from __future__ import annotations

import argparse
import logging
import math
import re
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np
import pdfplumber
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - exercised only when tqdm is absent.
    tqdm = None


DEFAULT_LABEL = "Stock"
DEFAULT_DPI = 250
JPEG_QUALITY = 94
SUPPORTED_EXTENSIONS = (".pdf",)
COMMON_POPPLER_DIRS = (
    Path(r"C:\poppler\poppler-26.02.0\Library\bin"),
    Path(r"C:\poppler\Library\bin"),
    Path(r"C:\poppler\bin"),
    Path(r"C:\tools\poppler\bin"),
    Path(r"C:\ProgramData\chocolatey\lib\poppler\tools\Library\bin"),
)


@dataclass(frozen=True)
class CaptionLocation:
    """Caption bounds in PDF coordinate space."""

    page_number: int
    x0: float
    x1: float
    top: float
    bottom: float
    page_width: float
    page_height: float

    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2.0


@dataclass(frozen=True)
class ImageCoordinates:
    """Caption bounds in rendered image coordinates."""

    x0: int
    x1: int
    top: int
    bottom: int

    @property
    def center_x(self) -> int:
        return (self.x0 + self.x1) // 2


@dataclass(frozen=True)
class Roi:
    """Region of interest in rendered image coordinates."""

    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height


@dataclass
class CandidateRectangle:
    """Detected photograph candidate in full rendered-image coordinates."""

    x: int
    y: int
    width: int
    height: int
    source: str
    score: float = 0.0

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2.0

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2.0

    @property
    def aspect_ratio(self) -> float:
        if self.height == 0:
            return 0.0
        return self.width / self.height


class ScraperError(Exception):
    """Recoverable per-PDF scraper error."""


class CaptionDetector:

    _TOKEN_PATTERN = re.compile(r"[^a-z0-9]+")

    @classmethod
    def find_all_captions(
        cls,
        pdf_path: Path,
        label: str
    ) -> dict[int, list[CaptionLocation]]:

        # The label is now supplied by CLI, so normalize it once into the same
        # token shape used for extracted PDF words.
        target = tuple(cls._normalize(w) for w in label.split())
        if not target:
            raise ScraperError("Caption label cannot be empty.")

        results: dict[int, list[CaptionLocation]] = {}

        with pdfplumber.open(pdf_path) as pdf:

            for page_number, page in enumerate(pdf.pages):

                words = page.extract_words(
                    x_tolerance=3,
                    y_tolerance=3,
                    keep_blank_chars=False,
                    use_text_flow=False,
                )

                matches = cls._find_word_sequences(
                    words,
                    target
                )

                if not matches:
                    continue

                page_results = []

                for match in matches:

                    page_results.append(
                        CaptionLocation(
                            page_number=page_number,
                            x0=min(w["x0"] for w in match),
                            x1=max(w["x1"] for w in match),
                            top=min(w["top"] for w in match),
                            bottom=max(w["bottom"] for w in match),
                            page_width=float(page.width),
                            page_height=float(page.height),
                        )
                    )

                results[page_number] = page_results

        return results

    @classmethod
    def page_image_boxes(
        cls,
        pdf_path: Path,
        page_number: int,
        image_size: tuple[int, int],
    ) -> list[CandidateRectangle]:
        """Return embedded PDF image boxes in rendered-image coordinates."""

        image_width, image_height = image_size
        boxes: list[CandidateRectangle] = []

        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_number]
            scale_x = image_width / float(page.width)
            scale_y = image_height / float(page.height)

            for image in page.images:
                x0 = round(float(image["x0"]) * scale_x)
                x1 = round(float(image["x1"]) * scale_x)
                top = round(float(image["top"]) * scale_y)
                bottom = round(float(image["bottom"]) * scale_y)
                boxes.append(
                    CandidateRectangle(
                        x=min(x0, x1),
                        y=min(top, bottom),
                        width=abs(x1 - x0),
                        height=abs(bottom - top),
                        source="pdf-image",
                    )
                )

        return boxes

    @classmethod
    def _find_word_sequences(
        cls,
        words: Sequence[dict[str, object]],
        target: Sequence[str],
    ) -> list[list[dict[str, object]]]:
        matches: list[list[dict[str, object]]] = []
        normalized = [cls._normalize(str(word.get("text", ""))) for word in words]
        target_length = len(target)

        for start in range(0, len(words) - target_length + 1):
            if tuple(normalized[start : start + target_length]) == tuple(target):
                matches.append(list(words[start : start + target_length]))

        return matches

    @classmethod
    def _normalize(cls, value: str) -> str:
        return cls._TOKEN_PATTERN.sub("", value.lower())


class PageRenderer:
    """Renders a single PDF page with pdf2image."""

    def __init__(self, dpi: int, poppler_path: Optional[str]) -> None:
        self.dpi = dpi
        self.poppler_path = poppler_path or self._detect_poppler_path()

    def render(self, pdf_path: Path, page_number: int) -> Image.Image:
        try:
            pages = convert_from_path(
                pdf_path,
                dpi=self.dpi,
                first_page=page_number + 1,
                last_page=page_number + 1,
                poppler_path=self.poppler_path,
                fmt="ppm",
                thread_count=1,
            )
        except PDFInfoNotInstalledError as exc:
            raise ScraperError(
                "Poppler is not installed or is not on PATH. Install Poppler "
                "or pass --poppler-path pointing to its bin directory."
            ) from exc
        except PDFPageCountError as exc:
            raise ScraperError(f"Unable to read PDF page count: {exc}") from exc

        if not pages:
            raise ScraperError("pdf2image returned no rendered pages.")
        return pages[0].convert("RGB")

    @staticmethod
    def _detect_poppler_path() -> Optional[str]:
        for folder in COMMON_POPPLER_DIRS:
            if (folder / "pdfinfo.exe").exists() and (folder / "pdftoppm.exe").exists():
                logging.info("Using Poppler at %s", folder)
                return str(folder)
        return None


class CoordinateConverter:
    """Converts pdfplumber coordinates into rendered image coordinates."""

    @staticmethod
    def caption_to_image(
        caption: CaptionLocation,
        image_size: tuple[int, int],
    ) -> ImageCoordinates:
        image_width, image_height = image_size
        scale_x = image_width / caption.page_width
        scale_y = image_height / caption.page_height
        return ImageCoordinates(
            x0=round(caption.x0 * scale_x),
            x1=round(caption.x1 * scale_x),
            top=round(caption.top * scale_y),
            bottom=round(caption.bottom * scale_y),
        )


class PhotographDetector:
    """Detects and ranks photograph rectangles in the ROI above the caption."""

    def __init__(self, debug: bool, debug_folder: Path) -> None:
        self.debug = debug
        self.debug_folder = debug_folder

    def detect_candidates(
        self,
        page_image: Image.Image,
        captions: Sequence[ImageCoordinates],
        embedded_candidates: Sequence[CandidateRectangle],
        pdf_stem: str,
        page_number: int,
    ) -> tuple[np.ndarray, list[CandidateRectangle], Roi]:
        """Detect all reusable photograph rectangles for one rendered page."""

        if not captions:
            raise ScraperError("No captions supplied for photograph detection.")

        cv_image = self._pil_to_cv(page_image)
        # Build one page-level ROI from all caption-specific ROIs so OpenCV
        # contour detection runs once per page instead of once per caption.
        roi = self._build_combined_roi(cv_image.shape, captions)
        roi_image = cv_image[roi.y : roi.y2, roi.x : roi.x2]

        candidates = self._detect_rendered_candidates(roi_image, roi)
        candidates.extend(self._filter_page_embedded_candidates(embedded_candidates, roi))
        candidates = self._merge_candidates(candidates)

        if not candidates:
            self._save_debug_images(pdf_stem, page_number, cv_image, roi, [], None, None)
            raise ScraperError("No photograph candidates detected for caption page.")

        for candidate in candidates:
            # Keep a stable, page-level ordering for logs before per-caption
            # scoring overwrites scores during selection.
            candidate.score = candidate.area

        self._save_debug_images(pdf_stem, page_number, cv_image, roi, candidates, None, None)
        return cv_image, candidates, roi

    def select_candidate(
        self,
        caption: ImageCoordinates,
        candidates: Sequence[CandidateRectangle],
        roi: Roi,
    ) -> tuple[int, CandidateRectangle, list[CandidateRectangle]]:
        """Score reusable page candidates and select the best one for a caption."""

        eligible = [
            candidate
            for candidate in candidates
            if candidate.y2 <= caption.top
            and candidate.x2 >= roi.x
            and candidate.x <= roi.x2
        ]
        if not eligible:
            raise ScraperError("No photograph candidates detected above caption.")

        scored: list[tuple[int, CandidateRectangle]] = []
        for index, candidate in enumerate(candidates, start=1):
            if candidate not in eligible:
                continue
            candidate.score = self._score_candidate(candidate, caption, roi)
            scored.append((index, candidate))

        scored.sort(key=lambda item: item[1].score, reverse=True)
        selected_index, selected = scored[0]
        return selected_index, selected, [candidate for _, candidate in scored]

    def crop_candidate(
        self,
        cv_image: np.ndarray,
        candidate: CandidateRectangle,
    ) -> tuple[Image.Image, CandidateRectangle]:
        """Trim the selected rectangle and return the final crop image."""

        best = self._trim_white_border(cv_image, candidate)
        if best.area <= 0:
            raise ScraperError("Best candidate collapsed after border trimming.")

        crop = cv_image[best.y : best.y2, best.x : best.x2]
        return self._cv_to_pil(crop), best

    def save_debug_selection(
        self,
        pdf_stem: str,
        page_number: int,
        caption_index: int,
        cv_image: np.ndarray,
        roi: Roi,
        candidates: Sequence[CandidateRectangle],
        selected: CandidateRectangle,
    ) -> None:
        """Write per-caption debug overlays without rerunning detection."""

        self._save_debug_images(
            pdf_stem,
            page_number,
            cv_image,
            roi,
            candidates,
            selected,
            caption_index,
        )

    def _build_roi(self, image_shape: tuple[int, int, int], caption: ImageCoordinates) -> Roi:
        image_height, image_width = image_shape[:2]
        caption_width = max(caption.x1 - caption.x0, 1)
        side_padding = max(caption_width * 2.8, image_width * 0.18)

        x = max(0, int(caption.center_x - side_padding))
        x2 = min(image_width, int(caption.center_x + side_padding))

        gap = max(8, int(image_height * 0.008))
        max_height = int(image_height * 0.42)
        min_y = max(0, caption.top - max_height)
        y2 = max(0, caption.top - gap)

        if y2 <= min_y:
            min_y = max(0, caption.top - max(80, int(image_height * 0.2)))
            y2 = max(min_y + 1, caption.top)

        return Roi(x=x, y=min_y, width=max(1, x2 - x), height=max(1, y2 - min_y))

    def _build_combined_roi(
        self,
        image_shape: tuple[int, int, int],
        captions: Sequence[ImageCoordinates],
    ) -> Roi:
        caption_rois = [self._build_roi(image_shape, caption) for caption in captions]
        x = min(roi.x for roi in caption_rois)
        y = min(roi.y for roi in caption_rois)
        x2 = max(roi.x2 for roi in caption_rois)
        y2 = max(roi.y2 for roi in caption_rois)
        return Roi(x=x, y=y, width=max(1, x2 - x), height=max(1, y2 - y))

    def _detect_rendered_candidates(
        self,
        roi_image: np.ndarray,
        roi: Roi,
    ) -> list[CandidateRectangle]:
        masks = [
            self._non_white_mask(roi_image),
            self._edge_mask(roi_image),
        ]
        candidates: list[CandidateRectangle] = []
        for mask in masks:
            candidates.extend(self._contours_to_candidates(mask, roi))
        return candidates

    def _non_white_mask(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = np.where((gray < 245) | (hsv[:, :, 1] > 24), 255, 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return mask

    def _edge_mask(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 40, 130)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        return cv2.dilate(edges, kernel, iterations=2)

    def _contours_to_candidates(
        self,
        mask: np.ndarray,
        roi: Roi,
    ) -> list[CandidateRectangle]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi_area = roi.width * roi.height
        candidates: list[CandidateRectangle] = []

        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            area = width * height
            if area < roi_area * 0.025:
                continue
            if area > roi_area * 0.92:
                continue
            if width < roi.width * 0.12 or height < roi.height * 0.12:
                continue

            aspect = width / max(height, 1)
            if not 0.45 <= aspect <= 2.8:
                continue

            candidates.append(
                CandidateRectangle(
                    x=roi.x + x,
                    y=roi.y + y,
                    width=width,
                    height=height,
                    source="rendered",
                )
            )
        return candidates

    def _filter_embedded_candidates(
        self,
        candidates: Sequence[CandidateRectangle],
        roi: Roi,
        caption: ImageCoordinates,
    ) -> list[CandidateRectangle]:
        filtered: list[CandidateRectangle] = []
        roi_area = roi.width * roi.height
        for candidate in candidates:
            if candidate.y2 > caption.top:
                continue
            if candidate.x2 < roi.x or candidate.x > roi.x2:
                continue
            if candidate.area < roi_area * 0.018:
                continue
            if candidate.area > roi_area * 0.95:
                continue
            if not 0.45 <= candidate.aspect_ratio <= 2.8:
                continue
            filtered.append(candidate)
        return filtered

    def _filter_page_embedded_candidates(
        self,
        candidates: Sequence[CandidateRectangle],
        roi: Roi,
    ) -> list[CandidateRectangle]:
        filtered: list[CandidateRectangle] = []
        roi_area = roi.width * roi.height
        for candidate in candidates:
            if candidate.x2 < roi.x or candidate.x > roi.x2:
                continue
            if candidate.y2 < roi.y or candidate.y > roi.y2:
                continue
            if candidate.area < roi_area * 0.006:
                continue
            if candidate.area > roi_area * 0.95:
                continue
            if not 0.45 <= candidate.aspect_ratio <= 2.8:
                continue
            filtered.append(candidate)
        return filtered

    def _merge_candidates(
        self,
        candidates: Sequence[CandidateRectangle],
    ) -> list[CandidateRectangle]:
        ordered = sorted(candidates, key=lambda item: item.area, reverse=True)
        merged: list[CandidateRectangle] = []

        for candidate in ordered:
            for index, current in enumerate(merged):
                if self._iou(candidate, current) >= 0.55:
                    merged[index] = self._union(current, candidate)
                    break
            else:
                merged.append(candidate)
        return merged

    def _score_candidate(
        self,
        candidate: CandidateRectangle,
        caption: ImageCoordinates,
        roi: Roi,
    ) -> float:
        horizontal_distance = abs(candidate.center_x - caption.center_x)
        horizontal_score = 1.0 - min(horizontal_distance / max(roi.width / 2.0, 1), 1.0)

        vertical_gap = max(caption.top - candidate.y2, 0)
        expected_gap = max(roi.height * 0.03, 1.0)
        vertical_score = math.exp(-vertical_gap / max(expected_gap * 4.0, 1.0))

        area_ratio = candidate.area / max(roi.width * roi.height, 1)
        area_score = self._triangular_score(area_ratio, low=0.08, ideal=0.34, high=0.80)

        aspect = candidate.aspect_ratio
        aspect_score = self._triangular_score(aspect, low=0.55, ideal=1.25, high=2.25)

        source_bonus = 0.06 if candidate.source == "pdf-image" else 0.0
        return (
            0.40 * horizontal_score
            + 0.26 * vertical_score
            + 0.20 * area_score
            + 0.14 * aspect_score
            + source_bonus
        )

    def _trim_white_border(
        self,
        image: np.ndarray,
        candidate: CandidateRectangle,
    ) -> CandidateRectangle:
        crop = image[candidate.y : candidate.y2, candidate.x : candidate.x2]
        if crop.size == 0:
            return candidate

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        mask = np.where((gray < 246) | (hsv[:, :, 1] > 18), 255, 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        points = cv2.findNonZero(mask)
        if points is None:
            return candidate

        x, y, width, height = cv2.boundingRect(points)
        pad = max(2, round(min(candidate.width, candidate.height) * 0.01))
        new_x = max(candidate.x + x - pad, candidate.x)
        new_y = max(candidate.y + y - pad, candidate.y)
        new_x2 = min(candidate.x + x + width + pad, candidate.x2)
        new_y2 = min(candidate.y + y + height + pad, candidate.y2)
        return CandidateRectangle(
            x=new_x,
            y=new_y,
            width=max(1, new_x2 - new_x),
            height=max(1, new_y2 - new_y),
            source=candidate.source,
            score=candidate.score,
        )

    def _save_debug_images(
        self,
        pdf_stem: str,
        page_number: int,
        page_image: np.ndarray,
        roi: Roi,
        candidates: Sequence[CandidateRectangle],
        selected: Optional[CandidateRectangle],
        caption_index: Optional[int],
    ) -> None:
        if not self.debug:
            return

        self.debug_folder.mkdir(parents=True, exist_ok=True)
        suffix = f"page_{page_number + 1}"
        if caption_index is not None:
            suffix = f"{suffix}_caption_{caption_index}"

        cv2.imwrite(
            str(self.debug_folder / f"{pdf_stem}_{suffix}_roi.jpg"),
            page_image[roi.y : roi.y2, roi.x : roi.x2],
        )

        overlay = page_image.copy()
        cv2.rectangle(overlay, (roi.x, roi.y), (roi.x2, roi.y2), (255, 0, 0), 3)
        for index, candidate in enumerate(candidates, start=1):
            color = (0, 255, 255)
            cv2.rectangle(
                overlay,
                (candidate.x, candidate.y),
                (candidate.x2, candidate.y2),
                color,
                3,
            )
            cv2.putText(
                overlay,
                f"{index}:{candidate.score:.2f}",
                (candidate.x, max(20, candidate.y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )
        if selected is not None:
            cv2.rectangle(
                overlay,
                (selected.x, selected.y),
                (selected.x2, selected.y2),
                (0, 255, 0),
                5,
            )
            crop = page_image[selected.y : selected.y2, selected.x : selected.x2]
            cv2.imwrite(str(self.debug_folder / f"{pdf_stem}_{suffix}_final_crop.jpg"), crop)

        cv2.imwrite(str(self.debug_folder / f"{pdf_stem}_{suffix}_boxes.jpg"), overlay)

    @staticmethod
    def _triangular_score(value: float, low: float, ideal: float, high: float) -> float:
        if value <= low or value >= high:
            return 0.0
        if value == ideal:
            return 1.0
        if value < ideal:
            return (value - low) / (ideal - low)
        return (high - value) / (high - ideal)

    @staticmethod
    def _iou(first: CandidateRectangle, second: CandidateRectangle) -> float:
        x1 = max(first.x, second.x)
        y1 = max(first.y, second.y)
        x2 = min(first.x2, second.x2)
        y2 = min(first.y2, second.y2)
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        union = first.area + second.area - intersection
        if union <= 0:
            return 0.0
        return intersection / union

    @classmethod
    def iou(cls, first: CandidateRectangle, second: CandidateRectangle) -> float:
        return cls._iou(first, second)

    @staticmethod
    def _union(first: CandidateRectangle, second: CandidateRectangle) -> CandidateRectangle:
        x = min(first.x, second.x)
        y = min(first.y, second.y)
        x2 = max(first.x2, second.x2)
        y2 = max(first.y2, second.y2)
        source = first.source if first.source == second.source else "merged"
        return CandidateRectangle(x=x, y=y, width=x2 - x, height=y2 - y, source=source)

    @staticmethod
    def _pil_to_cv(image: Image.Image) -> np.ndarray:
        return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _cv_to_pil(image: np.ndarray) -> Image.Image:
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


class PDImageScraper:
    """Batch processor for PDFs with caption-linked images."""

    def __init__(
        self,
        input_folder: Path,
        output_folder: Path,
        debug_folder: Path,
        dpi: int,
        poppler_path: Optional[str],
        debug: bool,
        overwrite: bool,
        label: str,
    ) -> None:
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.debug_folder = debug_folder
        self.debug = debug
        self.overwrite = overwrite
        self.label = label
        self.label_slug = self._slugify_label(label)
        self.renderer = PageRenderer(dpi=dpi, poppler_path=poppler_path)
        self.detector = PhotographDetector(debug=debug, debug_folder=debug_folder)

    def process(self) -> int:
        self._ensure_folders()
        pdfs = self._list_pdfs()
        if not pdfs:
            logging.warning("No PDF files found in %s", self.input_folder)
            return 0

        processed = 0
        failed = 0
        skipped = 0

        for pdf_path in self._progress(pdfs):
            output_dir = self.output_folder / pdf_path.stem
            if self._has_existing_outputs(output_dir) and not self.overwrite:
                logging.info("Skipping %s because output already exists", pdf_path.name)
                skipped += 1
                continue

            try:
                self._process_one(pdf_path, output_dir)
                processed += 1
            except ScraperError as exc:
                failed += 1
                logging.warning("%s skipped: %s", pdf_path.name, exc)
            except Exception as exc:  # pragma: no cover - defensive batch guard.
                failed += 1
                logging.error("%s failed unexpectedly: %s", pdf_path.name, exc)
                logging.debug(traceback.format_exc())

        logging.info(
            "Done. Processed=%s Skipped=%s Failed=%s Total=%s",
            processed,
            skipped,
            failed,
            len(pdfs),
        )
        return 0 if failed == 0 else 1

    def _process_one(self, pdf_path: Path, output_dir: Path) -> None:
        logging.info("Processing %s", pdf_path.name)

        captions_by_page = CaptionDetector.find_all_captions(pdf_path, self.label)
        caption_count = sum(len(captions) for captions in captions_by_page.values())
        if caption_count == 0:
            raise ScraperError(f'Caption "{self.label}" not found.')

        logging.info("Found %s %s captions", caption_count, self.label)
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.overwrite:
            for existing in output_dir.glob(f"{self.label_slug}_*.jpg"):
                existing.unlink()

        saved_rectangles: list[CandidateRectangle] = []
        saved_count = 0
        caption_index = 0

        for page_number, captions in captions_by_page.items():
            logging.info("Rendering page %s", page_number + 1)
            page_image = self.renderer.render(pdf_path, page_number)
            image_size = page_image.size
            caption_coords = [
                CoordinateConverter.caption_to_image(caption, image_size)
                for caption in captions
            ]
            embedded = CaptionDetector.page_image_boxes(pdf_path, page_number, image_size)
            cv_image, candidates, roi = self.detector.detect_candidates(
                page_image=page_image,
                captions=caption_coords,
                embedded_candidates=embedded,
                pdf_stem=pdf_path.stem,
                page_number=page_number,
            )
            logging.info("Detected %s photograph candidates", len(candidates))

            for caption_coords_item in caption_coords:
                caption_index += 1
                selected_index, selected, scored = self.detector.select_candidate(
                    caption=caption_coords_item,
                    candidates=candidates,
                    roi=roi,
                )
                logging.info("Caption %s -> Candidate %s", caption_index, selected_index)
                crop, final_rectangle = self.detector.crop_candidate(cv_image, selected)

                if any(
                    PhotographDetector.iou(final_rectangle, saved) > 0.80
                    for saved in saved_rectangles
                ):
                    logging.info(
                        "Skipping duplicate crop for caption %s (IoU > 0.80)",
                        caption_index,
                    )
                    continue

                saved_count += 1
                saved_rectangles.append(final_rectangle)
                self.detector.save_debug_selection(
                    pdf_stem=pdf_path.stem,
                    page_number=page_number,
                    caption_index=caption_index,
                    cv_image=cv_image,
                    roi=roi,
                    candidates=scored,
                    selected=final_rectangle,
                )
                output_path = output_dir / f"{self.label_slug}_{saved_count:03d}.jpg"
                crop.save(output_path, "JPEG", quality=JPEG_QUALITY, optimize=True)
                logging.info("Saved %s", output_path.name)

        if saved_count == 0:
            raise ScraperError("All selected photograph candidates were duplicates.")

    def _ensure_folders(self) -> None:
        if not self.input_folder.exists():
            raise ScraperError(f"Input folder does not exist: {self.input_folder}")
        self.output_folder.mkdir(parents=True, exist_ok=True)
        if self.debug:
            self.debug_folder.mkdir(parents=True, exist_ok=True)

    def _list_pdfs(self) -> list[Path]:
        return sorted(
            path
            for path in self.input_folder.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    def _has_existing_outputs(self, output_dir: Path) -> bool:
        if not output_dir.exists():
            return False
        return any(output_dir.glob(f"{self.label_slug}_*.jpg"))

    @staticmethod
    def _slugify_label(label: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
        return slug or "caption"

    @staticmethod
    def _progress(paths: Sequence[Path]) -> Iterable[Path]:
        if tqdm is None:
            return paths
        return tqdm(paths, desc="Processing PDFs", unit="pdf")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PDF Caption Image Extractor")
    parser.add_argument(
        "-i",
        "--input",
        default="input_pdfs",
        help="Folder containing PDF files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output",
        help="Folder where extracted JPG files are saved.",
    )
    parser.add_argument(
        "--debug-folder",
        default="debug",
        help="Folder where debug ROI/box/crop images are saved.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save ROI, bounding-box, and final-crop debug images.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help="Render DPI for the caption page.",
    )
    parser.add_argument(
        "--poppler-path",
        default=None,
        help="Optional path to Poppler bin directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output JPG files.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--label",
        default=DEFAULT_LABEL,
        help="Caption label to extract.",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def main() -> int:
    args = parse_arguments()
    configure_logging(args.log_level)

    scraper = PDImageScraper(
        input_folder=Path(args.input),
        output_folder=Path(args.output),
        debug_folder=Path(args.debug_folder),
        dpi=args.dpi,
        poppler_path=args.poppler_path,
        debug=args.debug,
        overwrite=args.overwrite,
        label=args.label,
    )
    return scraper.process()


if __name__ == "__main__":
    raise SystemExit(main())
