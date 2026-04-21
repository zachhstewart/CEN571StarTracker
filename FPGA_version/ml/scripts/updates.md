Input: Is the same images.
Output:
The current model outputs two things simultaneously, vs a direction-only model that would output just one:

Current dual-head model outputs:

cls_logits — 6 raw scores, one per sky sector (coarse direction, 90° apart)
\\
quat — [w, x, y, z] unit quaternion (precise 3D orientation, continuous)
