from typing import Any, Dict, Union

import torch
from transformers import Trainer


class MyTrainer(Trainer):

    overwrite_inputs_callback = None

    def _prepare_inputs(
        self, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = super()._prepare_inputs(inputs)
        if self.overwrite_inputs_callback is not None:
            inputs = self.overwrite_inputs_callback(inputs)

        return inputs
