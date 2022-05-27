""" model_statistics.py """
from typing import Any, List, Tuple

from .formatting import FormattingOptions
from .layer_info import LayerInfo, prod


class ModelStatistics:
    """Class for storing results of the summary."""

    def __init__(
        self,
        summary_list: List[LayerInfo],
        input_size: Any,
        total_input_size: int,
        formatting: FormattingOptions,
    ) -> None:
        self.summary_list = summary_list
        self.input_size = input_size
        self.formatting = formatting
        self.total_input = total_input_size
        self.total_params, self.trainable_params = 0, 0
        self.total_output, self.total_mult_adds = 0, 0
        self.total_output_mem = 0 # debugging
        self.params_mem_per_layer_str = ''
        self.output_mem_per_layer_str = ''

        for layer_info in summary_list:
            if layer_info.is_leaf_layer:
                self.total_mult_adds += layer_info.macs
                if layer_info.is_recursive:
                    continue
                self.total_params += layer_info.num_params
                self.trainable_params += layer_info.trainable_params
                if layer_info.num_params > 0:
                    # x2 for gradients
                    self.total_output += 2 * prod(layer_info.output_size)
                    self.total_output_mem += layer_info.output_mem

                    self.params_mem_per_layer_str += (
                        f"{layer_info.get_layer_name(True, False)}: "
                        f"{self.float_to_megabytes(layer_info.num_params):.2}\n"
                    )
                    self.output_mem_per_layer_str += (
                        f"{layer_info.get_layer_name(True, False)}: "
                        f"{layer_info.output_mem}\n"
                    )
        print(f'[DEBUG][model_statistics.py] ModelStatistics.__init__ - Total output memory size for forward pass (MB): {round(self.total_output_mem, 2)}')
        self.formatting.set_layer_name_width(summary_list)

    def __repr__(self) -> str:
        """Print results of the summary."""
        divider = "=" * self.formatting.get_total_width()
        summary_str = (
            f"{divider}\n"
            f"{self.formatting.header_row()}{divider}\n"
            f"{self.formatting.layers_to_str(self.summary_list)}{divider}\n"
            f"Total params: {self.total_params:,}\n"
            f"Trainable params: {self.trainable_params:,}\n"
            f"Non-trainable params: {self.total_params - self.trainable_params:,}\n"
        )
        if self.input_size:
            summary_str += (
                "Total mult-adds ({}): {:0.2f}\n{}\n"  # pylint: disable=consider-using-f-string  # noqa
                "Input size (MB): {:0.2f}\n"
                "Forward/backward pass size (MB): {:0.2f}\n"
                "Params size (MB): {:0.2f}\n"
                "Estimated Total Size (MB): {:0.2f}\n{}\n"
                "Parameter size per layer (MB)\n{}{}\n"
                "Output size per layer (MB)\n{}".format(
                    *self.to_readable(self.total_mult_adds),
                    divider,
                    self.to_megabytes(self.total_input),
                    self.float_to_megabytes(self.total_output),
                    self.float_to_megabytes(self.total_params),
                    (
                        self.to_megabytes(self.total_input)
                        + self.float_to_megabytes(self.total_output + self.total_params)
                    ),
                    divider,
                    self.params_mem_per_layer_str,
                    divider,
                    self.output_mem_per_layer_str
                )
            )
        summary_str += divider
        return summary_str

    @staticmethod
    def float_to_megabytes(num: int) -> float:
        """Converts a number (assume floats, 4 bytes each) to megabytes."""
        return num * 4 / (1024 ** 2)

    @staticmethod
    def to_megabytes(num: int) -> float:
        """Converts a number (assume floats, 4 bytes each) to megabytes."""
        return num / (1024 ** 2)

    @staticmethod
    def to_readable(num: int) -> Tuple[str, float]:
        """Converts a number to millions, billions, or trillions."""
        if num >= (1024 ** 4):
            return "T", num / (1024 ** 4)
        if num >= (1024 ** 3):
            return "G", num / (1024 ** 3)
        return "M", num / (1024 ** 2)
