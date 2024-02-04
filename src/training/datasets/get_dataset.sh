URL=https://efrosgans.eecs.berkeley.edu/cyclegan/datasets/monet2photo.zip
ZIP_FILE=./datasets/monet2photo.zip
TARGET_DIR=./datasets/monet2photo/
wget -N $URL -O $ZIP_FILE
mkdir $TARGET_DIR
unzip $ZIP_FILE -d ./datasets/
rm $ZIP_FILE