#!/usr/bin/env python3

import argparse
import numpy as np
import cv2
import os
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter


def vibrance_filter(image, strength, smoothness=0.0, skin_tone_protection=True, skin_hue_center=10, skin_hue_width=30):
    """
    Vibrance Filter - Enhanced version with strength and smoothness controls
    Photoshop-like vibrance implementation with alpha channel support
    
    Args:
        image: Input image (BGR or BGRA format)
        strength: Vibrance strength (0-1+)
        smoothness: Saturation smoothing (0-1, default: 0)
        skin_tone_protection: Whether to protect skin tones
        skin_hue_center: Center hue for skin tones (0-180 for OpenCV)
        skin_hue_width: Width of skin tone protection zone
    
    Returns:
        Processed image (same format as input)
    """
    
    # Check if image has alpha channel
    has_alpha = image.shape[2] == 4
    alpha_channel = None
    
    if has_alpha:
        # Separate alpha channel
        alpha_channel = image[:, :, 3]
        image = image[:, :, :3]
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    
    # Normalize values
    h = h / 180.0 * 360  # Convert to 0-360 degrees
    s = s / 255.0 * 100  # Convert to 0-100%
    v = v / 255.0
    
    # Step 1: Create saturation enhancement curve (Vibrance core)
    saturation_points = np.array([0, 20, 50, 80, 100])
    enhancement_curve = np.array([
        0,
        20 + 15 * strength,
        50 + 20 * strength,  
        80 + 10 * strength,
        100
    ])
    
    curve_spline = CubicSpline(saturation_points, enhancement_curve)
    s_modified = curve_spline(s)
    
    # Step 2: Apply saturation smoothing if needed
    if smoothness > 0:
        # Convert smoothness from 0-1 to sigma range (0-5 pixels)
        sigma = smoothness * 5.0
        s_modified = gaussian_filter(s_modified, sigma=sigma)
    
    # Step 3: Create skin tone protection mask
    if skin_tone_protection:
        skin_center = skin_hue_center * 2
        skin_width = skin_hue_width
        
        hue_diff = np.abs(h - skin_center)
        hue_diff = np.minimum(hue_diff, 360 - hue_diff)
        
        skin_mask = np.clip((hue_diff - skin_width/3) / (skin_width/3), 0, 1)
        core_skin_mask = (hue_diff < skin_width/2).astype(np.float32)
        skin_protection = 0.2 + 0.8 * (1 - core_skin_mask)
        protection_mask = skin_protection * skin_mask + (1 - skin_mask) * 0.2
    else:
        protection_mask = np.ones_like(s)
    
    # Step 4: Apply final saturation with blending
    s_final = s * (1 - protection_mask) + s_modified * protection_mask
    s_final = np.clip(s_final, 0, 100)
    
    # Optional: Apply additional overall smoothing to final saturation
    if smoothness > 0.3:  # Only apply strong smoothing if smoothness is high
        additional_sigma = (smoothness - 0.3) * 3.0
        s_final = gaussian_filter(s_final, sigma=additional_sigma)
    
    # Convert back to OpenCV format
    h_final = h / 360.0 * 180
    s_final = s_final / 100.0 * 255
    v_final = v * 255
    
    hsv_final = np.stack([h_final, s_final, v_final], axis=2).astype(np.uint8)
    result = cv2.cvtColor(hsv_final, cv2.COLOR_HSV2BGR)
    
    # Restore alpha channel if exists
    if has_alpha:
        result = np.dstack([result, alpha_channel])
    
    return result


def grow_opaque_regions(bgr, alpha, iterations=2, kernel_size=3):
    threshold = 0  # можно настроить
    valid_mask = (alpha > threshold).astype(np.uint8) * 255

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    valid_mask_eroded = cv2.erode(valid_mask, kernel, iterations=iterations)
    
    inpaint_mask = (valid_mask_eroded == 0).astype(np.uint8) * 255

    bgr_blurred = cv2.medianBlur(bgr, 3)
    bgr_inpainted = cv2.inpaint(bgr_blurred, inpaint_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    return bgr_inpainted


def fix_alpha_edges(img):
    bgr = img[:, :, :3]
    alpha = img[:, :, 3]
    
    grown = grow_opaque_regions(bgr, alpha)

    alpha_median = cv2.medianBlur(alpha, 3)

    kernel = np.ones((3,3), np.uint8)
    alpha_eroded = cv2.erode(alpha, kernel)
    
    alpha_float = alpha_eroded.astype(float) / 255.0
    alpha_3ch = cv2.merge([alpha_float, alpha_float, alpha_float])
    
    smoothed_edges = (
        bgr * alpha_3ch + 
        grown * (1.0 - alpha_3ch)
    ).astype(np.uint8)

    black = np.zeros_like(bgr)
    alpha_float_median = alpha_median.astype(float) / 255.0
    alpha_3ch_median = cv2.merge([alpha_float_median, alpha_float_median, alpha_float_median])

    pma_edges = (
        smoothed_edges * alpha_3ch_median + 
        black * (1.0 - alpha_3ch_median)
    ).astype(np.uint8)

    alpha_median_eroded = cv2.erode(alpha_median, kernel)
    alpha_fin = cv2.addWeighted(alpha_median, 0.5, alpha_median_eroded, 0.5, 0)

    return cv2.merge([pma_edges, alpha_median_eroded])


def load_image_with_alpha(image_path):
    """
    Load image with alpha channel support
    """
    # Read image with alpha channel
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if image is None:
        return None
    
    # Handle different image formats
    if image.shape[2] == 4:
        # BGRA format - already good
        return image
    elif image.shape[2] == 3:
        # BGR format - add solid alpha channel
        height, width = image.shape[:2]
        alpha_channel = np.ones((height, width), dtype=image.dtype) * 255
        return np.dstack([image, alpha_channel])
    else:
        # Grayscale or other - convert to BGRA
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        height, width = image_bgr.shape[:2]
        alpha_channel = np.ones((height, width), dtype=image_bgr.dtype) * 255
        return np.dstack([image_bgr, alpha_channel])


def save_image_with_alpha(image, output_path):
    cv2.imwrite(output_path, image)


def main():
    parser = argparse.ArgumentParser(description='Apply vibrance filter to image')
    parser.add_argument('input_image', nargs='?', default='nst_result.png', help='Input image path')
    parser.add_argument('-o', '--output', help='Output image path (optional)')
    parser.add_argument('--no-skin-protection', action='store_true',
                       help='Disable skin tone protection')
    parser.add_argument('-s', '--strength', type=float, default=1.0,
                       help='Vibrance strength (0-2+, default: 1.0)')
    parser.add_argument('--smoothness', type=float, default=0.0,
                       help='Saturation smoothness (0-1, default: 0)')
    parser.add_argument('--skin-hue', type=int, default=10,
                       help='Skin tone hue center (0-180, default: 10)')
    parser.add_argument('--skin-width', type=int, default=30,
                       help='Skin tone protection width (default: 30)')
    parser.add_argument('--no-preserve-alpha', action='store_true',
                       help='Do not preserve alpha channel')
    
    args = parser.parse_args()
    
    # Validate parameters
    if args.strength < 0:
        print("Warning: Strength should be >= 0")
    
    if args.smoothness < 0 or args.smoothness > 1:
        print("Warning: Smoothness should be between 0 and 1")
        args.smoothness = max(0, min(1, args.smoothness))
    
    # Read input image with alpha support
    if not os.path.exists(args.input_image):
        print(f"Error: Input image '{args.input_image}' not found")
        return
    
    if not args.no_preserve_alpha:
        image = load_image_with_alpha(args.input_image)
    else:
        image = cv2.imread(args.input_image)
    
    if image is None:
        print(f"Error: Could not read image '{args.input_image}'")
        return
    
    has_alpha = image.shape[2] == 4
    print(f"Applying vibrance filter ({args.strength=}, {args.smoothness=}, {has_alpha=})...")
    
    # Apply vibrance filter
    result = vibrance_filter(
        image, 
        strength=args.strength,
        smoothness=args.smoothness,
        skin_tone_protection=not args.no_skin_protection,
        skin_hue_center=args.skin_hue,
        skin_hue_width=args.skin_width
    )

    if has_alpha:
        result = fix_alpha_edges(result)
    
    # Save result
    if args.output:
        output_path = args.output
    else:
        # Create default output filename
        name, ext = os.path.splitext(args.input_image)
        output_path = f"{name}_vibrance_s{args.strength}_sm{args.smoothness}{ext}"
    
    if not args.no_preserve_alpha:
        save_image_with_alpha(result, output_path)
    else:
        cv2.imwrite(output_path, result[:, :, :3])
    
    print(f"Result saved to: {output_path}")

if __name__ == "__main__":
    main()