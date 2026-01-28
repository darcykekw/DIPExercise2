import cv2
import numpy as np

def main():
    # part a: 4-neighborhood (n4) pixel relationship
    image_path = 'Clerigo_Grayscale.jpg'
    target_row = 150
    target_col = 150

    print(f"--- DIP Exercise 2 | Part A: N4 Neighborhood ---")
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load '{image_path}'.")
        return

    rows, cols = img.shape
    ref_value = img[target_row, target_col]
    print(f"Reference Pixel @ ({target_row}, {target_col}) | Intensity: {ref_value}")
    print("-" * 50)
    
    # n4 offsets
    n4_coords = {
        'Top': (-1, 0), 'Bottom': (1, 0),
        'Left': (0, -1), 'Right': (0, 1)
    }
    
    print("N4 Neighbors:")
    for name, (dr, dc) in n4_coords.items():
        r, c = target_row + dr, target_col + dc
        if 0 <= r < rows and 0 <= c < cols:
            val = img[r, c]
            print(f"  {name:<10}: ({r}, {c}) = {val}")
        else:
            print(f"  {name:<10}: Out of Bounds")

    # output visualization
    print("\n[3x3 Neighborhood Matrix Values]")
    roi = img[target_row-1:target_row+2, target_col-1:target_col+2]
    
    for i in range(3):
        row_str = ""
        for j in range(3):
            val = roi[i, j]
            dr, dc = i - 1, j - 1 
            is_n4 = (abs(dr) + abs(dc)) == 1
            is_center = (dr == 0 and dc == 0)
            
            if is_center:
                row_str += f"[{val:^3}] "
            elif is_n4:
                row_str += f" *{val:^3}  "
            else:
                row_str += f"  {val:^3}  "
        print(row_str)
    
    print("\nLegend: [ ] = Reference, * = N4 Neighbor")

    # save output image
    save_output_image(roi, n4_coords.values(), "Exer2_PartA_Output.png")

def save_output_image(roi, neighbor_offsets, filename):
    scale = 100
    h, w = roi.shape
    # create valid color image
    img_out = np.zeros((h * scale, w * scale, 3), dtype=np.uint8)
    
    offsets_list = list(neighbor_offsets)

    for i in range(h):
        for j in range(w):
            val = roi[i, j]
            # background color based on pixel intensity
            c_val = int(val)
            top_left = (j * scale, i * scale)
            bottom_right = ((j + 1) * scale, (i + 1) * scale)
            
            cv2.rectangle(img_out, top_left, bottom_right, (c_val, c_val, c_val), -1)
            
            # determine relationship
            dr, dc = i - 1, j - 1
            is_center = (dr == 0 and dc == 0)
            is_neighbor = False
            for (nr, nc) in offsets_list:
                if dr == nr and dc == nc:
                    is_neighbor = True
                    break
            
            # draw borders
            if is_center:
                cv2.rectangle(img_out, top_left, bottom_right, (0, 0, 255), 4) # red
            elif is_neighbor:
                cv2.rectangle(img_out, top_left, bottom_right, (255, 0, 0), 4) # blue
            else:
                cv2.rectangle(img_out, top_left, bottom_right, (128, 128, 128), 1) # gray
                
            # draw text
            text = str(val)
            scale_font = 1.0
            thickness = 2
            (t_w, t_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale_font, thickness)
            
            text_x = top_left[0] + (scale - t_w) // 2
            text_y = top_left[1] + (scale + t_h) // 2
            
            # text color contrast
            txt_col = (255, 255, 255) if val < 128 else (0, 0, 0)
            cv2.putText(img_out, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, scale_font, txt_col, thickness)

    cv2.imwrite(filename, img_out)
    print(f"Output image saved as: {filename}")

if __name__ == "__main__":
    main()
