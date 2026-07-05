from pathlib import Path

import Framework

extension_dir = Path(__file__).parent
__extension_name__ = extension_dir.name
__install_command__ = [
    "pip",
    "install",
    str(extension_dir),
    "--no-build-isolation",  # to build the extension using the current environment instead of creating a new one
]

try:
    from .FasterGSFastCudaBackend.torch_bindings.rasterization import (
        diff_rasterize,
        rasterize,
        rasterize_with_buffers,
        update_pruning_scores,
        count_metric_from_buffers,
        RasterizerSettings,
    )
    from .FasterGSFastCudaBackend.torch_bindings.adam import FusedAdam
    from .FasterGSFastCudaBackend.torch_bindings.filter3d import update_3d_filter
    from .FasterGSFastCudaBackend.torch_bindings.densification import (
        relocation_adjustment,
        add_noise,
    )

    __all__ = [
        "diff_rasterize",
        "rasterize",
        "rasterize_with_buffers",
        "update_pruning_scores",
        "count_metric_from_buffers",
        "RasterizerSettings",
        "FusedAdam",
        "update_3d_filter",
        "relocation_adjustment",
        "add_noise",
    ]
except ImportError as e:
    raise Framework.ExtensionError(
        name=__extension_name__, install_command=__install_command__
    )
