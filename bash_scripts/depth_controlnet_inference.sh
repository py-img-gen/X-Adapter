python inference.py --plugin_type "controlnet" \
--prompt "A colorful lotus, ink, high quality, extremely detailed" \
--condition_type "depth" \
--input_image_path "./assets/Lotus.jpeg" \
--controlnet_condition_scale_list 1.0 \
--adapter_guidance_start_list 0.80 \
--adapter_condition_scale_list 1.0 \
--height 1024 \
--width 1024 \
--height_sd1_5 512 \
--width_sd1_5 512 \
