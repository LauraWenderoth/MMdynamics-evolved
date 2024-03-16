import einops
from torch.utils.data import Dataset
from torchvision import transforms
from openslide import OpenSlide
import os
from multiprocessing import Lock
from multiprocessing import Manager
import h5py
import torch
import pprint
from einops import rearrange, repeat
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import *
from box import Box
import pandas as pd
import random


class TCGADataset(Dataset):

    def __init__(self, dataset: str,
                 path: str,
                 level: int  =  None,
                 filter_overlap: bool = True,
                 survival_analysis: bool = True,
                 num_classes: int = 2,
                 n_bins: int = 4,
                 sources: List = ["omic", "slides"],
                 log_dir = None,
                 ):
        """
        Dataset wrapper to load different TCGA data modalities (omic and WSI data).
        Args:
            dataset (str): TCGA dataset to load (e.g. "brca", "blca")
            config (Box): Config object
            filter_overlap: filter omic data and/or slides that do not have a corresponding sample in the other modality
            n_bins: number of discretised bins for survival analysis
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset = dataset
        self.log_dir = log_dir
        self.sources = sources
        self.filter_overlap = filter_overlap
        self.survival_analysis = survival_analysis
        self.sample_missing = False
        self.num_classes = num_classes
        self.n_bins = n_bins
        self.path = Path(path)
        self.subset = 'uncensored'
        self.raw_path = Path(self.path).joinpath(f"wsi/{dataset}")
        prep_path = Path(self.path).joinpath(f"wsi/{dataset}")
        self.prep_path = prep_path
        # create patch feature directory for first-time run
        os.makedirs(self.prep_path.joinpath("patch_features"), exist_ok=True)
        self.slide_ids = [slide_id.rsplit(".", 1)[0] for slide_id in os.listdir(prep_path)]
        # /net/archive/export/tcga/tcga/wsi/brca_preprocessed_level2/patches


        # for early fusion baseline, we need to concatenate omic and slide features into a single tensor
        self.concat = False

        valid_sources = ["omic", "slides", "rna-sequence", "mutation", "copy-number"]
        assert all([source in valid_sources for source in sources]), f"Invalid source specified. Valid sources are {valid_sources}"
        assert not ("omic" in list(self.sources) and any(option in list(self.sources) for option in ["rna-sequence", "mutation", "copy-number"])), f'Choose only "omic" or  "rna-sequence", "mutation", "copy-number"'

        self.wsi_paths: dict = self._get_slide_dict() # {slide_id: path}
        self.sample_slide_id = self.slide_ids[0] + ".svs"
        self.sample_slide = OpenSlide(self.wsi_paths[self.sample_slide_id])
        # pre-load and transform omic data
        omic_df = self.load_omic()
        self.omic_df = omic_df.copy()
        omic_df = omic_df.drop(["site", "oncotree_code", "case_id", "slide_id", "train", "censorship", "survival_months", "y_disc"],axis=1)
        self.omic_features = self.split_omics(omic_df)
        self.override = True
        self.omic_tensor = {key: torch.Tensor(self.omic_features[key].values) for key in self.omic_features.keys()}

        self.level = level
        self.slide_idx: dict = self._get_slide_idx() # {idx (molecular_df): slide_id}
        self.wsi_width, self.wsi_height = (256,256)
        self.censorship = self.omic_df["censorship"].values
        self.survival_months = self.omic_df["survival_months"].values
        self.y_disc = self.omic_df["y_disc"].values

        manager = Manager()
        self.patch_cache = manager.dict()
        print(f"Dataloader initialised for {dataset} dataset")
        self.get_info(full_detail=False)

    def __getitem__(self, index):
        y_disc = self.y_disc[index]
        censorship = self.censorship[index]
        event_time = self.survival_months[index]
        data = []

        if "omic" in self.sources:
            if any(source in ["rna-sequence", "mutation", "copy-number"] for source in self.sources):
                print(f'Only "omic" is used. Any of the following source will NOT be used: "rna-sequence", "mutation", "copy-number".')
            omic_tensor = self.omic_tensor['omic'][index]
            if self.concat:
                omic_tensor = torch.flatten(omic_tensor)
            data.append(omic_tensor)
        else:
            if "rna-sequence" in self.sources:
                omic_tensor = self.omic_tensor['rna-sequence'][index]
                if self.concat:
                    omic_tensor = torch.flatten(omic_tensor)
                data.append(omic_tensor)
            if "mutation" in self.sources:
                omic_tensor = self.omic_tensor['mutation'][index]
                if self.concat:
                    omic_tensor = torch.flatten(omic_tensor)
                data.append(omic_tensor)
            if "copy-number" in self.sources:
                omic_tensor = self.omic_tensor['copy-number'][index]
                if self.concat:
                    omic_tensor = torch.flatten(omic_tensor)
                data.append(omic_tensor)

        if "slides" in self.sources:
            slide_id = self.omic_df.iloc[index]["slide_id"].rsplit(".", 1)[0]



            if index not in self.patch_cache:
                slide_tensor = self.load_patch_features(slide_id)

                self.patch_cache[index] = slide_tensor

            else:
                slide_tensor = self.patch_cache[index]

            if self.concat:
                slide_tensor = torch.flatten(slide_tensor)
            data.append(slide_tensor)
        assert data, "You must select at least one source."

        if self.concat: # for early fusion baseline
            concat_tensor = torch.cat(data, dim=0)
            return [concat_tensor], censorship, event_time, y_disc
        else:
            return data, censorship, event_time, y_disc

    def _get_slide_idx(self):
        # filter slide index to only include samples with WSIs availables
        filter_keys = [slide_id + ".svs" for slide_id in self.slide_ids]
        tmp_df = self.omic_df[self.omic_df.slide_id.isin(filter_keys)]
        return dict(zip(tmp_df.index, tmp_df["slide_id"]))

    def __len__(self):
        if self.sources == ["omic"]:
            # use all omic samples when running single modality
            return self.omic_df.shape[0]
        else:
            # only use overlap otherwise
            return len(self.slide_ids)
    def _get_slide_dict(self):
        """
        Given the download structure of the gdc-client, each slide is stored in a folder
        with a non-meaningful name. This function returns a dictionary of slide_id to
        the path of the slide.
        Returns:
            svs_dict (dict): Dictionary of slide_id to path of slide
        """
        slide_path = Path(self.path).joinpath(f"wsi/{self.dataset}")
        svs_files = list(slide_path.glob("**/*.svs"))
        svs_dict = {path.name: path for path in svs_files}
        return svs_dict

    def _load_patch_coords(self):
        """
        Loads all patch coordinates for the dataset and level specified in the config and writes it to a dictionary
        with key: slide_id and value: patch coordinates (where each coordinate is a x,y tupe)
        """
        coords = {}
        for slide_id in self.slide_ids:
            patch_path = self.prep_path.joinpath(f"patches/{slide_id}.h5")
            h5_file = h5py.File(patch_path, "r")
            patch_coords = h5_file["coords"][:]
            coords[slide_id] = patch_coords
        return coords

    # @property
    # def sample_slide_id(self):
    #     return next(iter(self.wsi_paths.keys()))

    def get_info(self, full_detail: bool = False):
        """
        Logging util to print some basic dataset information. Normally called at the start of a pipeline run
        Args:
            full_detail (bool): Print all slide properties

        Returns:
            None
        """
        #slide_path = Path(self.config.tcga_path).joinpath(f"wsi/{self.dataset}/")
        print(f"Dataset: {self.dataset.upper()}")
        print(f"Molecular data shape: {self.omic_df.shape}")
        sample_overlap = (set(self.omic_df["slide_id"]) & set(self.wsi_paths.keys()))
        print(f"Molecular/Slide match: {len(sample_overlap)}/{len(self.omic_df)}")
        #TODO change print if subset of omics used
        print(f"Slide level count: {self.sample_slide.level_count}")
        print(f"Slide level dimensions: {self.sample_slide.level_dimensions}")
        print(f"Slide resize dimensions: w: {self.wsi_width}, h: {self.wsi_height}")
        print(f"Sources selected: {self.sources}")
        print(f"Censored share: {np.round(len(self.omic_df[self.omic_df['censorship'] == 1])/len(self.omic_df), 3)}")
        print(f"Survival_bin_sizes: {dict(self.omic_df['y_disc'].value_counts().sort_values())}")



    def show_samples(self, n=1):
        """
        Logging util to show some detailed sample stats and render the whole slide image (e.g., in a notebook)
        Args:
            n (int): Number of samples to show

        Returns:
            None
        """
        # sample_df = self.omic_df.sample(n=n)
        sample_df = self.omic_df[self.omic_df["slide_id"].isin(self.wsi_paths.keys())].sample(n=n)
        for idx, row in sample_df.iterrows():
            print(f"Case ID: {row['case_id']}")
            print(f"Patient age: {row['age']}")
            print(f"Gender: {'female' if row['is_female'] else 'male'}")
            print(f"Survival months: {row['survival_months']}")
            print(f"Survival years:  {np.round(row['survival_months']/12, 1)}")
            print(f"Censored (survived follow-up period): {'yes' if row['censorship'] else 'no'}")
            # print(f"Risk: {'high' if row['high_risk'] else 'low'}")
            # plot wsi
            slide, slide_tensor = self.load_wsi(row["slide_id"], level=self.level)
            print(f"Shape:", slide_tensor.shape)
            plt.figure(figsize=(10, 10))
            plt.imshow(slide_tensor)
            plt.show()




    def load_omic(self,
                  eps: float = 1e-6
                  ) -> pd.DataFrame:
        """
        Loads in omic data and returns a dataframe and filters depending on which whole slide images
        are available, such that only samples with both omic and WSI data are kept.
        Also calculates the discretised survival time for each sample.
        Args:
            eps (float): Epsilon value to add to min and max survival time to ensure all samples are included

        Returns:
            pd.DataFrame: Dataframe with omic data and discretised survival time (target)
        """
        data_path = Path(self.path).joinpath(f"omic/tcga_{self.dataset}_all_clean.csv.zip")
        df = pd.read_csv(data_path, compression="zip", header=0, index_col=0, low_memory=False)

        # handle missing values
        num_nans = df.isna().sum().sum()
        nan_counts = df.isna().sum()[df.isna().sum() > 0]
        df = df.fillna(df.mean(numeric_only=True))
        print(f"Filled {num_nans} missing values with mean")
        print(f"Missing values per feature: \n {nan_counts}")

        # filter samples for which there are no slides available
        if self.filter_overlap:
            slides_available = self.slide_ids
            omic_available = [id[:-4] for id in df["slide_id"]]
            overlap = set(slides_available) & set(omic_available)
            print(f"Slides available: {len(slides_available)}")
            print(f"Omic available: {len(omic_available)}")
            print(f"Overlap: {len(overlap)}")
            if len(slides_available) < len(omic_available):
                print(f"Filtering out {len(omic_available) - len(slides_available)} samples for which there are no omic data available")
                overlap_filter = [id + ".svs" for id in overlap]
                df = df[df["slide_id"].isin(overlap_filter)]
            elif len(slides_available) > len(omic_available):
                print(f"Filtering out {len(slides_available) - len(omic_available)} samples for which there are no slides available")
                self.slide_ids = overlap
            else:
                print("100% modality overlap, no samples filtered out")

        label_col = "survival_months"
        if self.subset == "all":
            df["y_disc"] = pd.qcut(df[label_col], q=self.n_bins, labels=False).values
        else:
            if self.subset == "censored":
                subset_df = df[df["censorship"] == 1]
            elif self.subset == "uncensored":
                subset_df = df[df["censorship"] == 0]
            # take q_bins from uncensored patients
            disc_labels, q_bins = pd.qcut(subset_df[label_col], q=self.n_bins, retbins=True, labels=False)
            q_bins[-1] = df[label_col].max() + eps
            q_bins[0] = df[label_col].min() - eps
            # use bin cuts to discretize all patients
            df["y_disc"] = pd.cut(df[label_col], bins=q_bins, retbins=False, labels=False, right=False,
                                  include_lowest=True).values

        df["y_disc"] = df["y_disc"].astype(int)

        if self.log_dir is not None:
            df.to_csv(self.log_dir.joinpath(f"{self.dataset}_omic_overlap.csv.zip"), compression="zip")

        return df

    def load_wsi(self, slide_id: str, level: int = None) -> Tuple:
        """
        Load in single slide and get region at specified resolution level
        Args:
            slide_id:
            level:
            resolution:

        Returns:
            Tuple (openslide object, tensor of region)
        """

        def transform_to_numpy(region):
            try:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x[:3, :, :]),  # remove alpha channel
                ])
                region_tensor = transform(region)
            except:
                transform = transforms.Compose([
                    transforms.ToTensor()
                ])
                region_tensor = transform(region)

            numpy_image = region_tensor.cpu().detach().numpy()
            numpy_image = numpy_image.transpose(1, 2, 0)
            return numpy_image
        slide = OpenSlide(self.raw_path.joinpath(f"{slide_id}.svs"))
        level = 0
        threshold_light = 0.75
        slide_width, slide_height = slide.level_dimensions[level]
        middle_width_start = (slide_width // 5) * 2
        middle_width_end = slide_width - middle_width_start
        region_width, region_height = (256, 256)
        middle_height_start = (slide_height // 5) * 1
        middle_height_end = slide_height - middle_height_start

        for i in range(10):
            rand_x = random.randint(middle_width_start, middle_width_end)
            rand_y = random.randint(middle_height_start, middle_height_end)
            region = slide.read_region((rand_x, rand_y), level, (region_width, region_height))
            gray = region.convert('L')
            gray_np = transform_to_numpy(gray)
            all_pixesl = 256 * 256
            white_pixels = np.sum(gray_np > threshold_light)
            procentage = white_pixels / all_pixesl
            if procentage > 0.5:
                break

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[:3, :, :]), # remove alpha channel
        ])

        region_tensor = transform(region)
        return slide, region_tensor

    def load_patch_features(self, slide_id: str) -> torch.Tensor:
        """
        Loads patch features for a single slide from torch.pt file
        Args:
            slide_id (str): Slide ID

        Returns:
            torch.Tensor: Patch features
        """
        load_path = self.prep_path.joinpath(f"patch_features/{slide_id}.pt")
        with open(load_path, "rb") as file:
            patch_features = torch.load(file, weights_only=True)
        return patch_features

    def split_omics(self,df):
        substrings = ['rnaseq', 'cnv', 'mut']
        dfs = [df.filter(like=sub) for sub in substrings]

        return {"omic":df, "rna-sequence":dfs[0],"mutation":dfs[2],"copy-number":dfs[1]}






if __name__ == '__main__':
    os.chdir("../../")
    data_path = '/net/archive/export/tcga/tcga'
    dataset='brca'
    '''
    slide_id ="TCGA-AQ-A04H-01Z-00-DX1.5AC1E459-EF27-401D-98FD-0AC16559AF17"
    raw_path= Path(data_path).joinpath(f"wsi/{dataset}")
    slide = OpenSlide(raw_path.joinpath(f"{slide_id}.svs"))
    for level in [1]:
        slide_width, slide_height = slide.level_dimensions[level]
        middle_width_start = (slide_width // 5)*2
        middle_width_end = slide_width - middle_width_start
        region_width, region_height = (256, 256)
        middle_height_start = (slide_height // 5)*2
        middle_height_end = slide_height - middle_height_start
        for i in range(10):
            rand_x = random.randint(middle_width_start, middle_width_end)
            rand_y = random.randint(middle_height_start, middle_height_end)
            region = slide.read_region((rand_x, rand_y), level, (region_width, region_height))
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x[:3, :, :]),  # remove alpha channel
                transforms.Resize((region_width, region_height)),
            ])

            region_tensor = transform(region)
            numpy_image = region_tensor.cpu().detach().numpy()

            # If your tensor has channels as the first dimension (C, H, W), transpose it
            if numpy_image.shape[0] == 3:
                numpy_image = numpy_image.transpose(1, 2, 0)

            # Plot the numpy array as an image
            plt.imshow(numpy_image)
            plt.axis('off')  # Turn off axis
            plt.show()

    brca = TCGADataset("brca", data_path, sources = ["rna-sequence", "mutation", "copy-number"])
    kirp = TCGADataset("kirp", data_path,level=0)
    blca = TCGADataset("blca", data_path,level=2)
    print(brca.omic_df.shape, kirp.omic_df.shape)
    slide, region_tensor = kirp.load_wsi("TCGA-B3-A6W5-01Z-00-DX1.96E31A69-6D56-4520-8AF3-C083FB49A7DB")
    slide, region_tensor = blca.load_wsi("TCGA-2F-A9KT-01Z-00-DX1.ADD6D87C-0CC2-4B1F-A75F-108C9EB3970F")
    slide, region_tensor = brca.load_wsi("TCGA-AQ-A04H-01Z-00-DX1.5AC1E459-EF27-401D-98FD-0AC16559AF17")
    print(type(slide),type(region_tensor))
    numpy_image = region_tensor.cpu().detach().numpy()

    # If your tensor has channels as the first dimension (C, H, W), transpose it
    if numpy_image.shape[0] == 3:
        numpy_image = numpy_image.transpose(1, 2, 0)

    # Plot the numpy array as an image
    plt.imshow(numpy_image)
    plt.axis('off')  # Turn off axis
    plt.show()
    '''


