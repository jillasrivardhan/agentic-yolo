
prompt =   """
    Analyze an image using YOLOv8 object detection.

    This tool takes the path of an image, runs YOLOv8 object detection,
    identifies the objects present in the image, and returns the detected
    objects along with their confidence scores and bounding box information.

    Args:
        image_path (str): The file path of the image to analyze.

    Returns:
        str: A summary of the objects detected in the image, including
        their class names and confidence scores.
    """