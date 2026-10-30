# The pipeline

Each case goes through the same steps, in a fixed order. Each step is done by a *stage*, which can be replaced by
another implementation (see the [config reference](config.md) and [writing a plugin](plugins.md)). This page
describes the default stages. FC is the femoral cartilage and TC the tibial cartilage; every step after
segmentation handles them separately.

![OAI analysis workflow](../doc_imgs/OAI_workflow.png)

| # | Step | Stage (config key) | Default | Reads | Produces |
|---|---|---|---|---|---|
| 1 | `preprocess` | `preprocessor` | `oai_default` | the input image | the image, in the atlas orientation, intensities 0–1 |
| 2 | `segment` | `segmenter` | `unet3d_oai` | 1 | a probability map per structure |
| 3 | `register` | `registerer` | `unigradicon` | 1, the atlas image | transforms between the image and the atlas |
| 4 | `mesh` | `mesh_generator` | `cuberille` | 2 | a surface per structure, in the image's space |
| 5 | `thickness` | `thickness_estimator` | `mesh_splitting` | 2, 3, 4 | the surface with a thickness at each point |
| 6 | `atlas_map` | `atlas_mapper` | `closest_point` | 3, 5 | the atlas surface with the case's thickness |
| 7 | `biomarkers/NAME` | `biomarker_extractors` | `thickness_map_2d` | 6 | the 2D thickness maps |

## 1. Preprocess

The image is read from a file or a DICOM folder. A right knee is mirrored so that it looks like a left one, and the
image is turned into the atlas' orientation. Intensities are then windowed: the 0.1th percentile becomes 0, the
99.9th becomes 1, and values outside are clipped. All later steps work on this image.

*Checks:* the axes point the same way as the atlas' (fail otherwise); the voxel spacing and image size are those
of the OAI sagittal 3D DESS scans (review otherwise).

## 2. Segment

A 3D U-Net trained on OAI knees runs over the image in overlapping patches and gives, for each voxel, the
probability that it is femoral or tibial cartilage.

*Checks:* each structure is found (fail otherwise); its volume is in the usual range; little of it is outside the
largest piece (FC) or two pieces (TC); the network is not too uncertain about it.

## 3. Register

uniGradICON, a registration network, is fine-tuned for 2 steps on the pair, then registers the image to the atlas
image. The result is two transforms: `phi_AB` resamples the image into the atlas' space, and `phi_BA` moves points
(such as surface vertices) from the image into atlas space.

*Checks:* the transform does not fold space; the warped image correlates with the atlas image.

## 4. Mesh

The cuberille algorithm builds each structure's surface where its probability is 0.5, as a triangle mesh in the
image's physical space.

*Check:* the surface is not empty (fail otherwise).

## 5. Thickness

Thickness is the distance between the inner surface of the cartilage (against the bone) and the outer one. To tell
them apart, the surface is moved into atlas space with `phi_BA`, small pieces farther than 10 mm from the atlas'
inner surface are dropped, and a smoothed copy is split into an inner and an outer half by the direction of its
faces. The two halves are then cut out of the original, unsmoothed surface in the image's space, and each vertex of
the outer half gets its distance to the inner half.

*Check:* little of the surface is thicker than 6 mm.

## 6. Atlas map

The thickness surface is moved into atlas space, and each vertex of the atlas' inner surface takes the thickness of
the closest point of the case's surface. Every case then has its thickness on the same set of points, which is what
makes cases comparable.

*Check:* little of the atlas surface is left without thickness.

## 7. Biomarkers: 2D thickness maps

The atlas surface has a fixed 2D layout: the femoral cartilage is unrolled around a cylinder, and the tibial
cartilage is projected onto a plane. The thickness is interpolated onto a 300 × 300 grid over that layout and saved
as an array (`FC_thickness.npy`) and a picture (`FC_thickness.png`). Grid points outside the cartilage are NaN
(about 40% of each map).

Biomarker extractors are a list: further ones can be added in the config or by a plugin.

## Where the results go

- **A project** keeps each step's files in its own numbered folder, with a `provenance.json` that records how they
  were made. See [projects](projects.md#a-cases-folder).
- **The single-image command** (`oai-analysis IMAGE OUT left`) writes everything into one folder.

### Files of the single-image command

The names are the ones the pipeline has always used, which knee-sarg relies on.

| File | Step | Content |
|---|---|---|
| `FC_thickness.npy`, `TC_thickness.npy` | 7 | 2D thickness maps (mm), NaN outside the cartilage |
| `FC_thickness.png`, `TC_thickness.png` | 7 | pictures of the maps |
| `in_image.nrrd` | 1 | the preprocessed image |
| `FC_prob.nrrd`, `TC_prob.nrrd` | 2 | probability maps |
| `resampling.tfm`, `modelling.tfm` | 3 | `phi_AB` and `phi_BA` |
| `FC_mesh_patient.vtk`, ... | 4 | surfaces |
| `FC_outer.vtk`, `FC_inner.vtk`, ... | 5 | outer surface with thickness, inner surface with its distance to the outer one |
| `FC_mesh_atlas.vtk`, `FC_mapped_mesh.vtk`, ... | 6 | the surface moved into atlas space; the atlas surface with the thickness |

With `--no_intermediates` only the step 7 files are written.
