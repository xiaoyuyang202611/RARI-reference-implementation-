"""Example template for `--adapter CUSTOM_BACKEND_TEMPLATE:MedicalRepairBackend`.
Fill this with your own legally obtained frozen pretrained segmentation model.
This deliberately does not invent any paper-specific checkpoints/parameters.
"""
from rari.backbone_adapter import RepairBackend


class MedicalRepairBackend(RepairBackend):
    def __init__(self):
        # Example: load the actually used segmentation backbone/checkpoint here.
        # Freeze weights & preprocessing for *all* test cases.
        raise NotImplementedError('Configure actual medical segmentation model/checkpoint')

    def execute(self,image,current_mask,action,case):
        # action: prompt, local, boundary, volume.
        # Return actual resulting mask in identical geometry to image.
        # Do not inspect case["gt"] or reference-mask-derived test prompts.
        raise NotImplementedError('Implement the original repair operations using real model inference')
