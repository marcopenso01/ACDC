{"nbformat":4,"nbformat_minor":0,"metadata":{"colab":{"name":"image_utils","version":"0.3.2","provenance":[],"collapsed_sections":[]},"kernelspec":{"name":"python3","display_name":"Python 3"}},"cells":[{"metadata":{"id":"uekibB8rTpGh","colab_type":"code","colab":{}},"cell_type":"code","source":["import numpy as np\n","from skimage import measure\n","import logging\n","\n","logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')\n","\n","try:\n","    import cv2\n","except:\n","    logging.warning('Could not import opencv. Augmentation functions will be unavailable.')\n","else:\n","    def rotate_image(img, angle, interp=cv2.INTER_LINEAR):\n","\n","        rows, cols = img.shape[:2]\n","        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)\n","        return cv2.warpAffine(img, rotation_matrix, (cols, rows), flags=interp)\n","\n","\n","    def resize_image(im, size, interp=cv2.INTER_LINEAR):\n","\n","        im_resized = cv2.resize(im, (size[1], size[0]), interpolation=interp)  # swap sizes to account for weird OCV API\n","        return im_resized\n","\n","\n","def convert_to_uint8(image):\n","    image = image - image.min()\n","    image = 255.0*np.divide(image.astype(np.float32),image.max())\n","    return image.astype(np.uint8)\n","\n","def normalise_image(image):\n","    '''\n","    make image zero mean and unit standard deviation\n","    '''\n","\n","    img_o = np.float32(image.copy())\n","    m = np.mean(img_o)\n","    s = np.std(img_o)\n","    return np.divide((img_o - m), s)\n","\n","def normalise_images(X):\n","    '''\n","    Helper for making the images zero mean and unit standard deviation i.e. `white`\n","    '''\n","\n","    X_white = np.zeros(X.shape, dtype=np.float32)\n","\n","    for ii in range(X.shape[0]):\n","\n","        Xc = X[ii,:,:,:]\n","        mc = Xc.mean()\n","        sc = Xc.std()\n","\n","        Xc_white = np.divide((Xc - mc), sc)\n","\n","        X_white[ii,:,:,:] = Xc_white\n","\n","    return X_white.astype(np.float32)\n","\n","\n","def reshape_2Dimage_to_tensor(image):\n","    return np.reshape(image, (1,image.shape[0], image.shape[1],1))\n","\n","\n","def keep_largest_connected_components(mask):\n","    '''\n","    Keeps only the largest connected components of each label for a segmentation mask.\n","    '''\n","\n","    out_img = np.zeros(mask.shape, dtype=np.uint8)\n","\n","    for struc_id in [1, 2, 3]:\n","\n","        binary_img = mask == struc_id\n","        blobs = measure.label(binary_img, connectivity=1)\n","\n","        props = measure.regionprops(blobs)\n","\n","        if not props:\n","            continue\n","\n","        area = [ele.area for ele in props]\n","        largest_blob_ind = np.argmax(area)\n","        largest_blob_label = props[largest_blob_ind].label\n","\n","        out_img[blobs == largest_blob_label] = struc_id\n","\n","return out_img"],"execution_count":0,"outputs":[]}]}