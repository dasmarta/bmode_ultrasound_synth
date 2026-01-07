import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

BASE_DIR = "data/s0914"
CT_PATH  = os.path.join(BASE_DIR, "ct.nii.gz")
SEG_DIR  = os.path.join(BASE_DIR, "segmentations")
files = os.listdir(SEG_DIR)

BONE_LABELS = [
    f.replace(".nii.gz", "")
    for f in files
    if (
        f.startswith("rib_") or
        f.startswith("vertebrae_") or
        f in [
            "femur_left", "femur_right",
            "hip_left", "hip_right",
            "humerus_left", "humerus_right",
            "scapula_left", "scapula_right",
            "clavicula_left", "clavicula_right",
            "sternum", "sacrum", "skull",
            "costal_cartilages"
        ]
    )
]

MUSCLE_LABELS = [
    f.replace(".nii.gz", "")
    for f in files
    if (
        "gluteus" in f or
        "iliopsoas" in f
    )
]

ORGAN_LABELS = [
    f.replace(".nii.gz", "")
    for f in files
    if (
        f.startswith("lung_") or
        f in [
        "liver","stomach","duodenum", "heart",
        "small_bowel","colon","pancreas","gallbladder",
        "esophagus","heart", "kidney_left", "kidney_right",
        "urinary_bladder", "prostate", "thyroid_gland",
        "adrenal_gland_left", "adrenal_gland_right","spleen"
        ]
    )
]

SOFT_TISSUE_LABELS = [
    f.replace(".nii.gz", "")
    for f in files
    if f.replace(".nii.gz", "") in [
        "aorta","inferior_vena_cava",
        "superior_vena_cava",
        "portal_vein_and_splenic_vein",
        "pulmonary_vein",
        "iliac_artery_left", "iliac_artery_right",
        "iliac_vena_left", "iliac_vena_right",
        "subclavian_artery_left", "subclavian_artery_right",
        "common_carotid_artery_left", "common_carotid_artery_right",
        "brachiocephalic_trunk","spinal_cord"
    ]
]

ct = nib.load(CT_PATH).get_fdata()
shape = ct.shape

bone_mask   = np.zeros(shape, dtype=bool)
muscle_mask = np.zeros(shape, dtype=bool)
soft_mask   = np.zeros(shape, dtype=bool)
fat_mask    = np.zeros(shape, dtype=bool)
organ_mask    = np.zeros(shape, dtype=bool)

def load_and_or(mask, name):
    path = os.path.join(SEG_DIR, name + ".nii.gz")
    if os.path.exists(path):
        return mask | (nib.load(path).get_fdata() > 0)
    return mask

for label in BONE_LABELS:
    bone_mask = load_and_or(bone_mask, label)

for label in MUSCLE_LABELS:
    muscle_mask = load_and_or(muscle_mask, label)

for label in ORGAN_LABELS:
    organ_mask = load_and_or(organ_mask, label)

for label in SOFT_TISSUE_LABELS:
    soft_mask = load_and_or(soft_mask, label)

# HU-based fat
fat_mask = (ct >= -190) & (ct <= -30)

z = shape[2] // 2

bone_2d   = bone_mask[:, :, z]
muscle_2d = muscle_mask[:, :, z]
organ_2d = organ_mask[:, :, z]
soft_2d   = soft_mask[:, :, z]
fat_2d    = fat_mask[:, :, z]

# ATTENUATION MAP
atten = np.zeros((shape[0], shape[1]), dtype=np.float32)

atten[fat_2d]    = 0.10
atten[soft_2d]   = 0.15
atten[muscle_2d] = 0.20
atten[organ_2d]  = 0.22
atten[bone_2d]   = 0.50


noise = np.random.normal(0, 0.02, atten.shape)
atten = np.clip(atten + noise, 0, 1) 

plt.imshow(atten, cmap="gray")
plt.title("Attenuation map")
plt.axis("off")
plt.savefig("attenuation_map_slice.png", dpi=300, bbox_inches="tight")
plt.show()