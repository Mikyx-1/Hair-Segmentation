from segment_anything import SamPredictor, sam_model_registry



sam = sam_model_registry["default"](checkpoint="/kaggle/input/sam-ckpt-1/sam_vit_h_4b8939.pth")
predictor = SamPredictor(sam)

image = cv2.imread("/kaggle/input/flickr-image-dataset/flickr30k_images/flickr30k_images/1000092795.jpg")[:, :, ::-1]
predictor.set_image(image)
masks, _, _ = predictor.predict(point_coords = np.array([[100, 100]]), 
                                point_labels = np.array([1]))

