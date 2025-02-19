import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.generate_annotations.AnnotationsGenerator import AnnotationsGenerator
import matplotlib.pyplot as plt


def compute_junction_distance(root_img_dir, root_ricepr_dir, histogram=False, percentile: int = None, mu_std=False, tallest_bin=False):
    """
    Compute the distance between junctions and returns a histogram (optional)
    the distance between junctions excludes the distance from one junction to end point

    Args:
        root_img_dir (str): The root image directory, consisting of African/ and Asian/
        root_ricepr_dir (str): The root ricepr directory, consisting of African/ and Asian/
        histogram (bool, optional): Defaults to False.
    """
    # Create a buffer to store all distance
    buffer = list()
    
    # Process African image
    img_names_afr = os.listdir(root_img_dir + "/African")
    for img in img_names_afr:
        if not img.endswith(".jpg"):
            continue
        img_path = root_img_dir + "/African/" + img
        ricepr_path = root_ricepr_dir + "/African/" + img[:-len(".jpg")] + ".ricepr"
        
        # Create a generator
        gen = AnnotationsGenerator(
            img_path=img_path,
            ricepr_path=ricepr_path
        )
        
        # Get junction distance
        junction_distance = gen.generate_junction_distance(return_distance=True)
        
        # Add to the buffer
        buffer += junction_distance
    
    # Process Asian image
    img_names_as = os.listdir(root_img_dir + "/Asian")
    for img in img_names_as:
        if not img.endswith(".jpg"):
            continue
        img_path = root_img_dir + "/Asian/" + img
        ricepr_path = root_ricepr_dir + "/Asian/" + img[:-len(".jpg")] + ".ricepr"
        
        # Create a generator
        gen = AnnotationsGenerator(
            img_path=img_path,
            ricepr_path=ricepr_path
        )
        
        # Get junction distance
        junction_distance = gen.generate_junction_distance(return_distance=True)
        
        # Add to the buffer
        buffer += junction_distance
        
    print(f"==>> We have {len(buffer)} junction distance values.")

    # Plot histogram
    if histogram:
        plt.figure(figsize=(8, 6))
        
        num_bins = int(2 * len(buffer) ** (1/3))  # rice rule
        n, bins, patches = plt.hist(buffer, bins=num_bins, color='skyblue', edgecolor='black')
        
        
        # Highlight the Q1 on the histogram
        if percentile:
            # Compute the percentile
            percentile_value = np.percentile(buffer, percentile)
            
            # Plot
            plt.axvline(percentile_value, color='blue', label=f'${percentile}$th percentile = {percentile_value:.2f}')
        
        # Highlight the mean and STD on the histogram
        if mu_std:
            # Compute the mean and standard deviation of the buffer
            mu = np.mean(buffer)
            sigma = np.std(buffer)
            print(f"==>> sigma: {sigma}")
            
            # Plot
            plt.axvline(mu, color='red', label=f'$\mu$ = {mu:.2f}')
            plt.axvline(mu - sigma, color='blue', label=f'$\mu-1\sigma$ = {mu - sigma:.2f}')
        

        # Highlight tallest bin
        if tallest_bin:
            # Find the tallest bin
            max_bin_index = np.argmax(n)  # Index of the tallest bin
            max_bin_height = n[max_bin_index]  # Height (count) of the tallest bin
            max_bin_range = (bins[max_bin_index], bins[max_bin_index + 1])  # Bin edges
            
            # Plot
            patches[max_bin_index].set_facecolor('red')

            # Annotate the bin range
            plt.text((max_bin_range[0] + max_bin_range[1]) / 2, max_bin_height, 
                    f"{max_bin_range[0]:.2f} - {max_bin_range[1]:.2f}", 
                    ha='center', va='bottom', fontsize=10, color='red', fontweight='medium')

        # Labels and title
        plt.xlabel("Edge length in pixels.")
        plt.ylabel("Frequency")
        plt.title("Histogram of edge lengths over 560 rice panicle images.")
        plt.legend()
        plt.show()
    
    
if __name__ == "__main__":
    compute_junction_distance(
        root_img_dir="data/raw",
        root_ricepr_dir="data/processed",
        histogram=True,
        percentile=20,
        mu_std=True,
        tallest_bin=False
    )