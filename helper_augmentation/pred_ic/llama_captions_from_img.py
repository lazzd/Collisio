
from typing import List, Dict, Any, Optional
from pathlib import Path
import base64
import time
import json
import logging
from together import Together
import sys
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('caption_generator.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class CaptionGenerator:
    """
    A class to generate descriptive captions for images using LLaMA Vision model.
    
    This class handles the entire pipeline of image caption generation:
    - Loading and encoding images
    - Making API calls to the LLaMA Vision model
    - Processing and saving the generated captions
    
    Attributes:
        client (Together): Instance of Together API client
        model_name (str): Name of the LLaMA model to use
        description_prompt (str): Prompt template for caption generation
    """

    def __init__(self, model_name: str = "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo") -> None:
        """
        Initialize the CaptionGenerator with specified model and API client.
        
        Args:
            model_name (str): Name of the LLaMA model to use
        
        Raises:
            ConnectionError: If unable to initialize Together client
        """
        try:
            self.client = Together()
            self.model_name = model_name
            logger.info(f"Initialized CaptionGenerator with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Together client: {str(e)}")
            raise ConnectionError(f"Failed to initialize Together client: {str(e)}")

        # Load prompt template
        self.description_prompt = """Generate exactly 15 different, precise, and descriptive captions for the provided image. Each caption must be unique and should vividly describe the scene, including relevant details such as objects, actions, colors, or context.

- **Output Format:** Return a Python list containing only the captions, with each caption as a string.
- **Strict Constraints:** DO NOT include any introductory text, explanations, or additional formatting outside the Python list structure. DO NOT include any other output after the list. 

Example output:
["Giraffes and zebras graze peacefully in the savannah under a golden sunset.", "A tall giraffe stands majestically, its long neck stretching towards the sky.", "Zebras with distinctive black-and-white stripes graze near the giraffes in the open grassland.", "The savannah stretches endlessly, dotted with sparse trees and distant hills.", "A baby giraffe watches its surroundings with curiosity, standing close to its mother.", "The warm hues of the setting sun cast a golden glow over the landscape.", "A herd of zebras grazes near a group of giraffes, creating a harmonious scene.", "The savannah is alive with the sounds of nature, blending the rustling of grass and distant calls.", "A giraffe bends down to nibble on the grass, while zebras graze nearby.", "The vast savannah is a canvas of green and gold, with the horizon fading into the distance.", "A giraffe stretches its neck to reach leaves on a tall tree, showcasing its grace.", "Zebras and giraffes coexist peacefully in the vast expanse of the savannah.", "The giraffe stands tall, its spotted coat contrasting with the green grass.", "A giraffe walks gracefully, its long legs moving with ease across the savannah.", "In the golden light of dusk, the herd of giraffes and zebras forms a mesmerizing scene."]
\nDo not include any additional text or output outside the Python list structure."""

    def encode_image(self, image_path: str) -> str:
        """
        Encode an image file to base64 string.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            str: Base64 encoded string of the image
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            IOError: If unable to read the image file
        """
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
                
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {str(e)}")
            raise

    def generate_chat_completion(self, base64_image: str) -> Any:
        """
        Generate image captions using the LLaMA Vision model.
        
        Args:
            base64_image (str): Base64 encoded image string
            
        Returns:
            Any: API response containing generated captions
            
        Raises:
            RuntimeError: If API call fails
        """
        try:
            return self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.description_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                }
                            },
                        ],
                    }
                ],
                stream=False,
            )
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            raise RuntimeError(f"Failed to generate captions: {str(e)}")

    def generate_captions(
        self, 
        json_path: str, 
        source_path: str, 
        output_path: str = './output_llama_captions_prompt_COOC.json', 
        max_images: int = 5181
    ) -> None:
        """
        Generate captions for a batch of images and save results to JSON.
        
        Args:
            json_path (str): Path to JSON file containing image metadata
            source_path (str): Base path to image directory
            output_path (str): Path to save generated captions
            max_images (int): Maximum number of images to process
            
        Raises:
            FileNotFoundError: If input JSON or image directory doesn't exist
            JSONDecodeError: If input JSON is invalid
            IOError: If unable to write output file
        """
        try:
            json_path = Path(json_path)
            source_path = Path(source_path)
            output_path = Path(output_path)
            
            if not json_path.exists():
                raise FileNotFoundError(f"JSON file not found: {json_path}")
            if not source_path.exists():
                raise FileNotFoundError(f"Source directory not found: {source_path}")
            
            with open(json_path) as f:
                data = json.load(f)
            
            logger.info(f"Starting caption generation for {min(len(data), max_images)} images")
            
            output_data = []
            for i in range(min(len(data), max_images)):
                try:
                    logger.info(f"Processing image {i+1}/{min(len(data), max_images)}")
                    
                    image_path = data[i]['image']
                    full_image_path = source_path / image_path
                    
                    base64_image = self.encode_image(str(full_image_path))
                    stream = self.generate_chat_completion(base64_image)
                    
                    raw_output = stream.choices[0].message.content
                    logger.info(f"Generated captions for image: {image_path}")

                    # Clean the output using regex
                    try:
                        # Find content between first '[' and last ']'
                        match = re.search(r'\[(.*)\]', raw_output, re.DOTALL)
                        if match:
                            cleaned_output = '[' + match.group(1) + ']'
                            cleared_outpur = cleaned_output.replace('"\"', "")
                            logger.info("Successfully extracted content between brackets")
                        else:
                            cleaned_output = raw_output
                            logger.warning("No list brackets found in output")
                    except Exception as e:
                        logger.warning(f"Regex cleaning failed: {str(e)}")
                        cleaned_output = raw_output
                    
                    try:
                        # Try to evaluate the string as a Python list
                        output = eval(cleaned_output)
                        if not isinstance(output, list):
                            logger.warning(f"Output is not a list for image {image_path}. Using raw output.")
                            output = cleaned_output
                    except Exception as e:
                        logger.warning(f"Failed to parse output as list for image {image_path}: {str(e)}")
                        output = cleaned_output
                    
                    output_sample = {
                        "image": image_path,
                        "caption": output
                    }
                    output_data.append(output_sample)
                    
                    # Save progress after each image
                    with open(output_path, 'w') as f:
                        json.dump(output_data, f, indent=2)
                        
                   
                except Exception as e:
                    logger.error(f"Failed to process image {i}: {str(e)}")
                    continue
                
        except Exception as e:
            logger.error(f"Caption generation failed: {str(e)}")
            raise


def main():
    """
    Main entry point of the script.
    """
    try:
        json_path = "./data_annotation/coco_test.json"
        source_path = "./data/datasets/coco/"
        
        logger.info("Starting caption generation process")
        caption_generator = CaptionGenerator()
        caption_generator.generate_captions(json_path, source_path)
        
    except Exception as e:
        logger.error(f"Program failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()