mkdir $1/mp4
ffmpeg -y -framerate 24 -pattern_type glob -i $1'/Generator_g/*.png' -c:v libx264 -pix_fmt yuv420p $1/mp4/Generator_g.mp4
ffmpeg -y -framerate 24 -pattern_type glob -i $1'/Generator_f/*.png' -c:v libx264 -pix_fmt yuv420p $1/mp4/Generator_f.mp4
ffmpeg -y -framerate 24 -pattern_type glob -i $1'/Discriminator_y/*.png' -c:v libx264 -pix_fmt yuv420p $1/mp4/Discriminator_y.mp4
ffmpeg -y -framerate 24 -pattern_type glob -i $1'/Discriminator_x/*.png' -c:v libx264 -pix_fmt yuv420p $1/mp4/Discriminator_x.mp4
ffmpeg -y -framerate 24 -pattern_type glob -i $1'/Generator_loss/*.png' -c:v libx264 -pix_fmt yuv420p $1/mp4/Generator_loss.mp4
ffmpeg -y -framerate 24 -pattern_type glob -i $1'/Discriminator_loss/*.png' -c:v libx264 -pix_fmt yuv420p $1/mp4/Discriminator_loss.mp4
ffmpeg -y -i $1/mp4/Generator_g.mp4 -i $1/mp4/Generator_f.mp4 \
-i $1/mp4/Discriminator_x.mp4 -i $1/mp4/Discriminator_y.mp4 \
-i $1/mp4/Generator_loss.mp4 -i $1/mp4/Discriminator_loss.mp4 \
-filter_complex "[1:v][0:v][4:v]hstack=inputs=3[top];[2:v][3:v][5:v]hstack=inputs=3[bottom];[top][bottom]vstack=inputs=2[v]" \
-map "[v]" $1/mp4/history.mp4

