mkdir $1/mp4
ffmpeg -y -framerate 2 -pattern_type glob -i $1'/cello_to_sax/*.png' -c:v libx264 -pix_fmt yuv420p $1/mp4/cello_to_sax.mp4
ffmpeg -y -framerate 2 -pattern_type glob -i $1'/sax_to_cello/*.png' -c:v libx264 -pix_fmt yuv420p $1/mp4/sax_to_cello.mp4
ffmpeg -y -framerate 2 -pattern_type glob -i $1'/Generator_loss/*.png' -c:v libx264 -pix_fmt yuv420p $1/mp4/Generator_loss.mp4
ffmpeg -y -framerate 2 -pattern_type glob -i $1'/Discriminator_loss/*.png' -c:v libx264 -pix_fmt yuv420p $1/mp4/Discriminator_loss.mp4
ffmpeg -y -i $1/mp4/cello_to_sax.mp4 -i $1/mp4/sax_to_cello.mp4 -i $1/mp4/Generator_loss.mp4 -i $1/mp4/Discriminator_loss.mp4 \
-filter_complex "[0:v][1:v]hstack=inputs=2[top];[2:v][3:v]hstack=inputs=2[bottom];[top][bottom]vstack=inputs=2[v]" \
-map "[v]" $1/mp4/history.mp4

