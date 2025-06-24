FOLDER = "hf_fine_tuned_render2photo_sdxl_lora_aspect_v4_final"
EPOCH = "15"
NOT_DESCALED_PREFIX = f"not descaled:"
DESCALED_PREFIX = f"descaled:"

def calculate_averages(filepath):
    try:
        # Read the file
        with open(filepath, 'r') as file:
            content = file.read()
        
        # Split content by lines for better parsing
        lines = content.split('\n')
        
        not_descaled_numbers = []
        descaled_numbers = []
        
        # Process each line
        for line in lines:
            line = line.strip()
            
            # Handle "not descaled:" line
            if line.startswith(NOT_DESCALED_PREFIX):
                numbers_text = line.replace(NOT_DESCALED_PREFIX, '').strip()
                # Parse numbers, removing any semicolons and filtering out invalid values
                numbers = []
                for num_str in numbers_text.split(','):
                    try:
                        cleaned = num_str.strip().rstrip(';')
                        if cleaned and cleaned.replace('.', '', 1).replace('-', '', 1).isdigit():
                            numbers.append(float(cleaned))
                    except ValueError:
                        continue
                not_descaled_numbers = numbers
                print(f"Found {len(numbers)} not descaled values")
            
            # Handle "descaled:" line (but not "not descaled:")
            elif line.startswith(DESCALED_PREFIX) and 'not descaled:' not in line:
                numbers_text = line.replace(DESCALED_PREFIX, '').strip()
                # Parse numbers, removing any semicolons and filtering out invalid values
                numbers = []
                for num_str in numbers_text.split(','):
                    try:
                        cleaned = num_str.strip().rstrip(';')
                        if cleaned and cleaned.replace('.', '', 1).replace('-', '', 1).isdigit():
                            numbers.append(float(cleaned))
                    except ValueError:
                        continue
                descaled_numbers = numbers
                print(f"Found {len(numbers)} descaled values")
        
        # Calculate averages
        not_descaled_avg = sum(not_descaled_numbers) / len(not_descaled_numbers) if not_descaled_numbers else 0
        descaled_avg = sum(descaled_numbers) / len(descaled_numbers) if descaled_numbers else 0
        combined_avg = (not_descaled_avg + descaled_avg) / 2 if (not_descaled_avg or descaled_avg) else 0
        
        # Print the results
        print(f"Average of not descaled values: {not_descaled_avg:.4f}")
        print(f"Average of descaled values: {descaled_avg:.4f}")
        print(f"Average of both: {combined_avg:.4f}")
        
        # Debug info
        print(f"\nDebug info:")
        print(f"Not descaled count: {len(not_descaled_numbers)}")
        print(f"Descaled count: {len(descaled_numbers)}")
        if not_descaled_numbers and descaled_numbers:
            print(f"First 5 not descaled: {not_descaled_numbers[:5]}")
            print(f"First 5 descaled: {descaled_numbers[:5]}")
            print(f"Data sets are different: {not_descaled_numbers[:5] != descaled_numbers[:5]}")
        
    except FileNotFoundError:
        print(f"File '{filepath}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Call the function
calculate_averages(f"{FOLDER}/training_losses_epoch_{EPOCH}.txt")