import logging
import os
import sys

from omegaconf import DictConfig, OmegaConf, ListConfig, open_dict
from omegaconf.errors import ConfigAttributeError
from pydantic import BaseModel as PydanticBaseModel
from data import Dataset
from helpers import flatten
from processing.differential_analysis import differential_comparison, \
    multi_group_compairson, time_course_analysis
from visualization.abundance_bars import run_plot_abundance_bars
from constants import assert_literal, data_files_keys_type
from visualization.isotopologue_proportions import run_isotopologue_proportions_plot
from visualization.mean_enrichment_line_plot import run_mean_enrichment_line_plot
from processing.pca_analysis import run_pca_analysis
from typing import Union
from visualization.pca_plot import run_pca_plot
import constants


logger = logging.getLogger(__name__)


class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True


class MethodConfig(BaseModel):
    label: str
    name: str

    def build(self) -> "Method":
        raise NotImplementedError


class AbundancePlotConfig(MethodConfig):
    """
    Sets default values or fills them from the method yaml file
    """

    barcolor: str = "timepoint"
    axisx: str = "condition"
    axisx_labeltilt: int = 20  # 0 is no tilt
    width_each_subfig: int = 3 # TODO: must be float
    wspace_subfigs: int = 1 # TODO : must be float

    def build(self) -> "AbundancePlot":
        return AbundancePlot(config=self)


class DifferentialAnalysisConfig(MethodConfig):
    """
    Sets default values or fills them from the method yaml file
    """

    grouping: ListConfig = ["condition", "timepoint"]
    qualityDistanceOverSpan: float
    correction_method: str = "bonferroni"
    impute_values: DictConfig

    def build(self) -> "DifferentialAnalysis":
        return DifferentialAnalysis(config=self)


class MultiGroupComparisonConfig(MethodConfig):
    """
    Sets default values or fills them from the method yaml file
    """

    grouping: ListConfig = ["condition", "timepoint"]
    correction_method: str = "bonferroni"
    impute_values: DictConfig

    def build(self) -> "MultiGroupComparison":
        return MultiGroupComparison(config=self)

      
class IsotopologueProportionsPlotConfig(MethodConfig):
    """
    Sets default values or fills them from the method yaml file
    """
    max_nb_carbons_possible: int = 24
    appearance_separated_time: bool = True
    separated_plots_by_condition: bool = False
    plots_height: float = 4.8
    sharey: bool =  False  # share y axis across subplots
    x_ticks_text_size: int = 18
    y_ticks_text_size: int = 19

    def build(self) -> "IsotopologueProportionsPlot":
        return IsotopologueProportionsPlot(config=self)


class MeanEnrichmentLinePlotConfig(MethodConfig):
    """
    Sets default values or fills them from the method yaml file
    """
    alpha: float = 1
    xaxis_title: str = "Time"
    color_lines_by: str = "condition"  # or  "metabolite"
    palette_condition: str = "paired"  # seaborn/matplotlib pals
    palette_metabolite: str = "auto_multi_color"  # or .csv path

    def build(self) -> "MeanEnrichmentLinePlot":
        return MeanEnrichmentLinePlot(config=self)


class PcaAnalysisConfig(MethodConfig):
    pca_split_further: Union[ListConfig, None] = ["timepoint"]

    def build(self) -> "PcaAnalysis":
        return PcaAnalysis(config=self)


class PcaPlotConfig(MethodConfig):
    pca_split_further: Union[ListConfig, None] = ["timepoint"]
    draw_ellipses: Union[str, None] = "condition"
    run_iris_demo: bool = False

    def build(self) -> "PcaPlot":
        return PcaPlot(config=self)


class TimeCourseAnalysisConfig(MethodConfig):
    grouping: ListConfig = ["condition", "timepoint"]
    qualityDistanceOverSpan: float
    correction_method: str = "bonferroni"
    impute_values: DictConfig

    def build(self) -> "TimeCourseAnalysis":
        return TimeCourseAnalysis(config=self)


class Method(BaseModel):
    config: MethodConfig

    def plot(self):
        logger.info("Will plot the method, with the following config: %s", self.config)

    def run(self, cfg: DictConfig, dataset: Dataset) -> None:
        logger.info("Not instantialted in the parent class.")
        raise NotImplementedError

    def check_expectations(self, cfg: DictConfig, dataset: Dataset) -> None:
        logger.info("Not instantialted in the parent class.")
        raise NotImplementedError


class AbundancePlot(Method):
    config: AbundancePlotConfig

    def run(self, cfg: DictConfig, dataset: Dataset) -> None:
        logger.info("Will plot the abundance plot, with the following config: %s", self.config)
        if not ("metabolites" in cfg.analysis.keys()):  # plotting for _all_ metabolites
            logger.warning("No selected metabolites provided, plotting for all; might result in ugly too wide plots")
            with open_dict(cfg):
                for c in set(dataset.metadata_df["short_comp"]):
                    cfg.analysis["metabolites"] = {c: dataset.abundances_df.index.to_list()}

        self.check_expectations(cfg, dataset)
        out_plot_dir = os.path.join(os.getcwd(), cfg.figure_path)
        os.makedirs(out_plot_dir, exist_ok=True)
        run_plot_abundance_bars(dataset, out_plot_dir, cfg)

    def check_expectations(self, cfg: DictConfig, dataset: Dataset) -> None:
        # check that necessary information is provided in the analysis config
        try:
            if not set(cfg.analysis.metabolites.keys()).issubset(dataset.metadata_df["short_comp"]):
                raise ValueError(
                    f"[Analysis > Metabolites > compartments] are missing from [Metadata > Compartments]"
                )
            if not set(cfg.analysis.timepoints).issubset(set(dataset.metadata_df["timepoint"])):
                raise ValueError(
                    f"[Analysis > Timepoints] time points provided in the config file are not present in [Metadata > timepoint]"
                )
        except ConfigAttributeError as e:
            logger.error(f"Mandatory parameter not provided in the config file:{e}, aborting")
            sys.exit(1)
        except ValueError as e:
            logger.error(f"Data inconsistency:{e}")
            sys.exit(1)


class DifferentialAnalysis(Method):
    config: DifferentialAnalysisConfig

    def run(self, cfg: DictConfig, dataset: Dataset) -> None:
        logger.info(f"The current working directory is {os.getcwd()}")
        logger.info("Current configuration is %s", OmegaConf.to_yaml(cfg))
        logger.info("Will perform differential analysis, with the following config: %s", self.config)
        out_table_dir = os.path.join(os.getcwd(), cfg.table_path)
        os.makedirs(out_table_dir, exist_ok=True)
        self.check_expectations(cfg, dataset)
        for file_name, test in cfg.analysis.statistical_test.items():
            if test is None:
                continue
            logger.info(f"Running differential analysis of {dataset.get_file_for_label(file_name)} using {test} test")
            differential_comparison(file_name, dataset, cfg, test, out_table_dir=out_table_dir)

    def check_expectations(self, cfg: DictConfig, dataset: Dataset) -> None:
        # check that necessary information is provided in the analysis config
        try:
            if not (all(len(c) == 2 for c in cfg.analysis.comparisons)):
                raise ValueError(
                    f"Number of conditions has to be 2 for a pairwise comparison, see config file"
                )

            if not all(any(item[0] in set(dataset.metadata_df["condition"]) for item in sublist)
                       for sublist in cfg.analysis.comparisons):
                raise ValueError(
                    f"Conditions provided for comparisons in the config file are not present in the metadata file, aborting"
                )
            if not all(any(item[1] in set(dataset.metadata_df["timepoint"]) for item in sublist)
                       for sublist in cfg.analysis.comparisons):
                raise ValueError(
                    f"Timepoints provided for comparisons in the config file are not present in the metadata file, aborting"
                )
            # all values in the comparisons arrays of arrays are in the metadata, either as conditions or timepoints
            conditions_and_tp = set(dataset.metadata_df["condition"]).union(set(dataset.metadata_df["timepoint"]))
            comparisons = set(flatten(cfg.analysis.comparisons))
            diff = comparisons.difference(conditions_and_tp)
            if not comparisons.issubset(conditions_and_tp):
                raise ValueError(
                    f"Comparisons > Conditions or timepoints provided in the config file {diff} are not present in the metadata file, aborting"
                )
            # comparison_mode is one of the constant values
        except ConfigAttributeError as e:
            logger.error(f"Mandatory parameter not provided in the config file:{e}, aborting")
            sys.exit(1)
        except ValueError as e:
            logger.error(f"Data inconsistency:{e}")
            sys.exit(1)

            
class MultiGroupComparison(Method):
    config: MultiGroupComparisonConfig

    def run(self, cfg: DictConfig, dataset: Dataset) -> None:
        logger.info(f"The current working directory is {os.getcwd()}")
        logger.info("Current configuration is %s", OmegaConf.to_yaml(cfg))
        logger.info("Will perform multi group analysis, with the following config: %s", self.config)
        out_table_dir = os.path.join(os.getcwd(), cfg.table_path)
        os.makedirs(out_table_dir, exist_ok=True)
        self.check_expectations(cfg, dataset)

        for file_name in cfg.analysis.datatypes:
            logger.info(f"Running multi group analysis of {dataset.get_file_for_label(file_name)}")
            multi_group_compairson(file_name, dataset, cfg, out_table_dir=out_table_dir)

    #TODO: add expectations on the compartementalised dfs?
    def check_expectations(self, cfg: DictConfig, dataset: Dataset) -> None:
        try:
            [assert_literal(dt, data_files_keys_type, "datatype") for dt in cfg.analysis.datatypes]

            if not set([sublist[0] for sublist in cfg.analysis.conditions]).issubset(set(dataset.metadata_df["condition"])):
                raise ValueError(
                    f"Conditions provided for comparisons in the config file are not present in the metadata file, aborting"
                )
            if not set([sublist[1] for sublist in cfg.analysis.conditions]).issubset(set(dataset.metadata_df["timepoint"])):
                raise ValueError(
                    f"Timepoints provided for comparisons in the config file are not present in the metadata file, aborting"
                )

        except ValueError as e:
            logger.error(f"Data inconsistency:{e}")


class IsotopologueProportionsPlot(Method):
    config: IsotopologueProportionsPlotConfig

    def run(self, cfg: DictConfig, dataset: Dataset) -> None:
        logger.info("Will perform isotopologue proportions stacked-bar plots, with the following config: %s", self.config)  

        if not (
                "metabolites" in cfg.analysis.keys()):  # plotting for _all_ metabolites
            logger.warning(
                "No selected metabolites provided, plotting for all may fail")
            with open_dict(cfg):
                compartments = list(set(dataset.metadata_df["short_comp"]))
                for c in compartments:
                    isotopologues_names = list(dataset.isotopologues_proportions_df.index.to_list())
                    metabolites = set(
                        [i.split("_m+")[0] for i in isotopologues_names]
                        )
                    cfg.analysis["metabolites"] = {c: list(metabolites)}

        self.check_expectations(cfg, dataset)
        out_plot_dir = os.path.join(os.getcwd(), cfg.figure_path)
        os.makedirs(out_plot_dir, exist_ok=True)
        run_isotopologue_proportions_plot(dataset, out_plot_dir, cfg)

    def check_expectations(self, cfg: DictConfig, dataset: Dataset) -> None:
        try:
            if not set(cfg.analysis.metabolites.keys()).issubset(dataset.metadata_df['short_comp']):
                raise ValueError(
                    f"[Analysis > Metabolites > compartments] are missing from [Metadata > Compartments]"
                )
            if not set(cfg.analysis.timepoints).issubset(
                    set(dataset.metadata_df["timepoint"])):
                raise ValueError(
                    f"[Analysis > Timepoints] time points provided in the config file are not present in [Metadata > timepoint]"
                )
            if not cfg.analysis.width_each_stack > float(0):
                raise ValueError(
                    f"[Analysis > width_each_stack] must be superior to 0"
                )
            if not cfg.analysis.wspace_stacks > float(0):
                raise ValueError(
                    f"[Analysis > wspace_stacks] must be superior to 0"
                )
            if not cfg.analysis.inner_numbers_size >= 0:
                raise ValueError(
                    f"[Analysis > wspace_stacks] must be greater or equal to 0"
                )
        except ConfigAttributeError as e:
            logger.error(
                f"Mandatory parameter not provided in the config file:{e}, aborting")
            sys.exit(1)
        except ValueError as e:
            logger.error(f"Data inconsistency: {e}")
            sys.exit(1)


class MeanEnrichmentLinePlot(Method):
    config: MeanEnrichmentLinePlotConfig

    def run(self, cfg: DictConfig, dataset: Dataset) -> None:
        logger.info("Will perform Mean Enrichment (syn. Fractional Contributions) line-plot with the following config: %s", self.config)
        if not("metabolites" in cfg.analysis.keys()):
            logger.warning(
                "No selected metabolites provided, plotting for all; might result in ugly too wide plots")
            with open_dict(cfg):
                cfg.analysis["metabolites"] = {}
                for c in set(dataset.metadata_df["short_comp"]):
                    cfg.analysis["metabolites"][c] = \
                          dataset.mean_enrichment_df.index.to_list()

        self.check_expectations(cfg, dataset)
        out_plot_dir = os.path.join(os.getcwd(), cfg.figure_path)
        os.makedirs(out_plot_dir, exist_ok=True)
        run_mean_enrichment_line_plot(dataset, out_plot_dir, cfg)

    def check_expectations(self, cfg: DictConfig, dataset: Dataset) -> None:
        try:
            if not set(cfg.analysis.metabolites.keys()).issubset(
                    dataset.metadata_df['short_comp']):
                raise ValueError(
                    f"[Analysis > Metabolites > compartments] are missing from [Metadata > Compartments]"
                )
            if not cfg.analysis.method.color_lines_by in ["metabolite", "condition"]:
                raise ValueError(
                    f"[config > analysis > method > color_lines_by] must be metabolite or condition"
                )
        except ConfigAttributeError as e:
            logger.error(
                f"Mandatory parameter not provided in the config file: {e}, aborting"
            )
            sys.exit(1)
        except ValueError as e:
            logger.error(f"Data inconsistency: {e}")
            sys.exit(1)


class PcaAnalysis(Method):
    config: PcaAnalysisConfig

    def run(self, cfg: DictConfig, dataset: Dataset) -> None:
        logger.info("Will perform PCA analysis and save tables, "
                    "with the following config: %s", self.config)
        out_table_dir = os.path.join(os.getcwd(), cfg.table_path)
        os.makedirs(out_table_dir, exist_ok=True)
        self.check_expectations(cfg, dataset)
        available_pca_suitable_datatypes = set(
            ['abundances', 'mean_enrichment']
        ).intersection(dataset.available_datasets)
        for file_name in available_pca_suitable_datatypes:
            logger.info(
                f"Running pca analysis of {dataset.get_file_for_label(file_name)}")
            run_pca_analysis(file_name, dataset, cfg,
                             out_table_dir, mode="save_tables")

    def check_expectations(self, cfg: DictConfig, dataset: Dataset) -> None:
        try:
            if (dataset.abundances_df is None) and (
                    dataset.mean_enrichment_df is None
            ):
                raise ValueError(
                    f"abundances and mean_enrichment are missing in [Dataset]"
                )
            if (cfg.analysis.method.pca_split_further is not None) and not (
                set(cfg.analysis.method.pca_split_further).issubset(
                    set(["condition", "timepoint"]))):
                raise ValueError(
                    f"Unknown parameters in [config > analysis > method]"
                )
        except ConfigAttributeError as e:
            logger.error(
                f"Mandatory parameter not provided in the config file: {e}, aborting"
            )
            sys.exit(1)
        except ValueError as e:
            logger.error(f"Data inconsistency: {e}")
            sys.exit(1)


class PcaPlot(Method):
    config: PcaPlotConfig

    def run(self, cfg: DictConfig, dataset: Dataset) -> None:
        logger.info("Will perform PCA plots and save figures, "
                    "with the following config: %s", self.config)
        out_plot_dir = os.path.join(os.getcwd(), cfg.figure_path)
        os.makedirs(out_plot_dir, exist_ok=True)
        self.check_expectations(cfg, dataset)
        available_pca_suitable_datatypes = set(
            ['abundances', 'mean_enrichment']
        ).intersection(dataset.available_datasets)
        for file_name in available_pca_suitable_datatypes:
            # call analysis:
            pca_results_dict = run_pca_analysis(
                file_name, dataset, cfg,
                out_table_dir="",  # no writing tables in PcaPlot Method
                mode="return_results_dict")
            # plot:
            logger.info(
                f"Running pca plot(s) of {dataset.get_file_for_label(file_name)}")
            run_pca_plot(pca_results_dict, cfg, out_plot_dir)

    def check_expectations(self, cfg: DictConfig, dataset: Dataset) -> None:
        try:
            if (dataset.abundances_df is None) and \
                    (dataset.mean_enrichment_df is None):
                raise ValueError(
                    f"abundances and mean_enrichment are missing in [Dataset]"
                )
            if (cfg.analysis.method.pca_split_further is not None) and not (
                set(cfg.analysis.method.pca_split_further).issubset(
                    set(["condition", "timepoint"]))):
                raise ValueError(
                    f"Unknown parameters in [config > analysis > method > "
                    f"pca_plot > pca_split_further]"
                )
            if (cfg.analysis.method.draw_ellipses is not None) and not (
                    cfg.analysis.method.draw_ellipses in
                    ["condition", "timepoint"]):
                raise ValueError(
                    f"Unknown parameters in [config > analysis > method > "
                    f"pca_plot > draw_ellipses]"
                )
        except ConfigAttributeError as e:
            logger.error(
                f"Mandatory parameter not provided in the config file: {e}, aborting"
            )
            sys.exit(1)
        except ValueError as e:
            logger.error(f"Data inconsistency: {e}")
            sys.exit(1)


class TimeCourseAnalysis(Method):
    def run(self, cfg: DictConfig, dataset: Dataset) -> None:
        logger.info(f"The current working directory is {os.getcwd()}")
        logger.info("Current configuration is %s", OmegaConf.to_yaml(cfg))
        logger.info(
            "Will perform time-course analysis, with the following config: %s",
            self.config)
        out_table_dir = os.path.join(os.getcwd(), cfg.table_path)
        os.makedirs(out_table_dir, exist_ok=True)
        self.check_expectations(cfg, dataset)
        for file_name, test in cfg.analysis.statistical_test.items():
            if test is None:
                continue
            logger.info(
                f"Running time-course analysis of {dataset.get_file_for_label(file_name)} using {test} test")
            time_course_analysis(file_name, dataset, cfg, test,
                                    out_table_dir=out_table_dir)

    def check_expectations(self, cfg: DictConfig, dataset: Dataset) -> None:
        user_tests = [t[1] for t in cfg.analysis.statistical_test.items()]

        try:
            if not set(user_tests).issubset(
                    set(constants.availtest_methods) )  :
                raise ValueError(
                    f"Statistical tests provided in the config file not recognized, aborting"
                )
            if not (
                    len(dataset.metadata_df['timepoint'].unique()) ==
                    len(dataset.metadata_df['timenum'].unique())
                    ) and (
                    len(dataset.metadata_df['timenum'].unique()) ==
                    len(list(set(list(zip(dataset.metadata_df['timenum'],
                                        dataset.metadata_df['timenum'])))))
                    ):
                raise ValueError(
                    f"Inconsistent timepoint and timenum columns in metadata"
                )

        except ConfigAttributeError as e:
            logger.error(
                f"Mandatory parameter not provided in the config file:{e}, aborting")
            sys.exit(1)
        except ValueError as e:
            logger.error(f"Data inconsistency:{e}")
            sys.exit(1)