import cv2
import numpy as np


class FoodQualityPipeline:
    """
    Kuru gıda mikroskobik görüntüleri için klasik görüntü işleme pipeline'ı.
    Adımlar: RGB->HSI -> Histogram Eşitleme -> Otsu Eşikleme -> Morfolojik İşlemler
    """

    @staticmethod
    def rgb_to_hsi(image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """RGB görüntüyü HSI (Hue, Saturation, Intensity) kanallarına ayırır."""
        img = image_rgb.astype(np.float32) / 255.0
        R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]

        # Intensity (Yoğunluk)
        I = (R + G + B) / 3.0

        # Saturation (Doygunluk)
        min_rgb = np.minimum(np.minimum(R, G), B)
        S = 1.0 - (3.0 / (R + G + B + 1e-6)) * min_rgb
        S = np.clip(S, 0, 1)

        # Hue (Renk Özü)
        num = 0.5 * ((R - G) + (R - B))
        den = np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + 1e-6
        theta = np.arccos(np.clip(num / den, -1.0, 1.0))

        H = np.where(B <= G, theta, 2 * np.pi - theta)
        H = H / (2 * np.pi)  # [0, 1] aralığına normalize et

        return H, S, I

    @staticmethod
    def enhance_intensity(intensity: np.ndarray) -> np.ndarray:
        """Intensity kanalı üzerinde Histogram Eşitleme uygular."""
        intensity_uint8 = np.clip(intensity * 255, 0, 255).astype(np.uint8)
        enhanced = cv2.equalizeHist(intensity_uint8)
        return enhanced

    @staticmethod
    def apply_otsu(enhanced_img: np.ndarray, blur_ksize: int = 5) -> np.ndarray:
        """Gürültüyü azaltmak için opsiyonel Gaussian Blur ve ardından Otsu Thresholding uygular."""
        if blur_ksize > 1:
            # Kernel boyutu tek sayı olmalı
            k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
            blurred = cv2.GaussianBlur(enhanced_img, (k, k), 0)
        else:
            blurred = enhanced_img

        _, binary_mask = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return binary_mask

    @staticmethod
    def apply_morphology(
        binary_mask: np.ndarray, open_ksize: int = 3, close_ksize: int = 5
    ) -> np.ndarray:
        """Küçük gürültüleri silmek için Opening, boşlukları doldurmak için Closing uygular."""
        kernel_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_ksize, open_ksize)
        )
        kernel_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_ksize, close_ksize)
        )

        opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_open)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)
        return closed

    def run(
        self,
        image_bgr: np.ndarray,
        blur_ksize: int = 5,
        open_ksize: int = 3,
        close_ksize: int = 5,
    ) -> dict[str, np.ndarray]:
        """Tüm pipeline'ı çalıştırır ve ara çıktıları sözlük olarak döndürür."""
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        _, _, intensity = self.rgb_to_hsi(image_rgb)
        enhanced = self.enhance_intensity(intensity)
        otsu_mask = self.apply_otsu(enhanced, blur_ksize=blur_ksize)
        final_mask = self.apply_morphology(
            otsu_mask, open_ksize=open_ksize, close_ksize=close_ksize
        )

        return {
            "original_rgb": image_rgb,
            "enhanced": enhanced,
            "final_mask": final_mask,
        }