# $1 = model_path, $2 = instrument_x, $3 = instrument_y

mkdir $1/mp4
ffmpeg -y -framerate 24 -pattern_type glob -i $1'/'$2'_to_'$3'/*.png' -c:v libx264 -pix_fmt yuv420p $1/mp4/x_to_y.mp4
ffmpeg -y -framerate 24 -pattern_type glob -i $1'/'$3'_to_'$2'/*.png' -c:v libx264 -pix_fmt yuv420p $1/mp4/y_to_x.mp4
ffmpeg -y -framerate 24 -pattern_type glob -i $1'/Generator_loss/*.png' -c:v libx264 -pix_fmt yuv420p $1/mp4/Generator_loss.mp4
ffmpeg -y -framerate 24 -pattern_type glob -i $1'/Discriminator_loss/*.png' -c:v libx264 -pix_fmt yuv420p $1/mp4/Discriminator_loss.mp4
ffmpeg -y -i $1/mp4/x_to_y.mp4 -i $1/mp4/y_to_x.mp4 -i $1/mp4/Generator_loss.mp4 -i $1/mp4/Discriminator_loss.mp4 \
-filter_complex "[0:v][1:v]hstack=inputs=2[top];[2:v][3:v]hstack=inputs=2[bottom];[top][bottom]vstack=inputs=2[v]" \
-map "[v]" $1/mp4/history.mp4

