import cv2

class ExtractImages:

    @staticmethod
    def extract(file_path_from: str,
                file_path_to: str,
                interval_msec: int
    ):
        video = cv2.VideoCapture(file_path_from)
        success, image = video.read()
        count = 0
        while success:
            cv2.imwrite(f"{file_path_to}/frame{count * interval_msec}ms.jpg", image)     # save frame as JPEG file
            video.set(cv2.CAP_PROP_POS_MSEC,(count * interval_msec))
            success, image = video.read()
            count += 1

