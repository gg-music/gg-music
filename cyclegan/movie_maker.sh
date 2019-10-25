mkdir $1/mp4
ffmpeg -y -framerate 2 -pattern_type glob -i $1'/cello_to_sax/*.png' -c:v libx264 -pix_fmt yuv420p $1/mp4/cello_to_sax.mp4
ffmpeg -y -framerate 2 -pattern_type glob -i $1'/sax_to_cello/*.png' -c:v libx264 -pix_fmt yuv420p $1/mp4/sax_to_cello.mp4
ffmpeg -y -i $1/mp4/cello_to_sax.mp4 -i $1/mp4/sax_to_cello.mp4  -filter_complex hstack=inputs=2 $1/mp4/cello_sax.mp4
