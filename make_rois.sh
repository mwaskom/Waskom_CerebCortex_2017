#! /bin/bash

# Make ROI masks
for roi in ifs ips pcc mspl; do
  make_masks.py -r yeo17_${roi} -label yeo17_${roi} -sample graymid -save_native
done

for roi in con fpn dan; do
  make_masks.py -r yeo17_${roi} -label yeo17_${roi} -sample graymid -save_native
done
