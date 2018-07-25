# Image_preprocessing_pipeline
Allows for multiple camera preprocessing  - will need to integrate with camera driver for stereo disparity maps

This is the general overview of the pipeline:
//dead pixel concealment using median filter
//black level compensation
//lens shading correction
//anti-aliasing noise noise filter
//awb gain control
//cfa interpolation
//gamma correction
//color correction
//color space conversion
//noise filter for chroma
//hue saturation control
//noise filter for luma
//edge enhancement
//contrast brightness control
//data formatter
