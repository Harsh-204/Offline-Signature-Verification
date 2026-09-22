# CEDAR Signature Dataset

## Download

Download the **CEDAR signature dataset** from the official source or a
mirror. The dataset contains 55 writers, each with 24 genuine and 24
forged signature images.

One common source:
<https://www.cedar.buffalo.edu/NIJ/Publications/Databases.html>

Download the CEDAR Signature Dataset from the specified source and place it in data/cedar/ according to the documented folder structure.

## Expected Folder Structure

After downloading, place the images so the layout looks like this:

```
data/
  cedar/
    full_org/           # genuine (original) signatures
      original_1_1.png
      original_1_2.png
      ...
      original_55_24.png
    full_forg/           # forged signatures
      forgeries_1_1.png
      forgeries_1_2.png
      ...
      forgeries_55_24.png
```

### Filename convention

- **Genuine**: `original_{writer}_{sample}.png` — writer ID 1‥55,
  sample 1‥24.
- **Forged**: `forgeries_{writer}_{sample}.png` — same numbering.

> **Do not rename files.** The data loader parses writer and sample IDs
> directly from the filenames.
