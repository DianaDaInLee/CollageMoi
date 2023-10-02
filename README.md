CollageMoi
================

The python function `CollageMoi` creates a collage based on a photo of
your choice (“main” photo) using all other “candidate” photos you choose
to be used to create the final collage. In short, it splits your main
photo into little tiles, identify which of the candidate photos have the
most similar RGB color scheme to each of the tiles, and stitch them back
together to create a collage.

### Arguments

- `fldr_main`: Name of the folder where the main photo is located
- `fldr_cand`: Name of the folder where all training photos are located
- `img_main`: Filename of the main photo
- `img_final`: Filename of the final collage photo you want to save as
- `nsplit_row`: (INTEGER) Number of rows of the main photo you want to
  split by
- `nsplit_col`: (INTEGER) Number of columns of the main photo you want
  to split by
- `space`: (INTEGER) White space between individual tiles in a main
  collage (default = 0)
- `dist_method`): Distance metric in comparing main photo against
  training photos (default = cv2.HISTCMP_INTERSECT). Choose from
  HISTCMP_CORREL, HISTCMP_CHISQR, HISTCMP_INTERSECT,
  HISTCMP_BHATTACHARYYA, HISTCMP_HELLINGER, HISTCMP_CHISQR_ALT,
  HISTCMP_KL_DIV.

For more information on RGB color similarity measures, see:
<https://docs.opencv.org/3.4/d6/dc7/group__imgproc__hist.html>.

![](Example/code_ex.png)

![](Example/ex.png)

Enjoy!
