for num in {1..2}
do
../../blender2.79/blender --background --python render_images.py -- --num_images 1 --use_gpu 1
clear
mv ../output/images/CLEVR_new_000000.png ../output/images/CLEVR_new_$num.png
mv ../output/scenes/CLEVR_new_000000.json ../output/images/CLEVR_new_$num.json
done

echo "Series of numbers from 1 to 10."
