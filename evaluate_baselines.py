import argparse
import csv
import json
import requests
import sys


def main():
    # Set up command-line arguments
    parser = argparse.ArgumentParser(
        description="Query captions from a CSV and output JSON results using a specified baseline."
    )
    parser.add_argument("input_file", help="Path to the input CSV file.")
    parser.add_argument("output_file", help="Path to the output JSON file.")
    parser.add_argument("baseline", help="Baseline model to use (e.g., 'clip', 'blip', 'flava', etc.).")
    args = parser.parse_args()

    results = []
    consecutive_errors = 0
    endpoint = "http://127.0.0.1:8020/retrieve/"

    # Open and read the CSV file
    with open(args.input_file, mode="r", newline='', encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        if "caption" not in reader.fieldnames:
            print("Error: 'caption' column not found in CSV file", file=sys.stderr)
            sys.exit(1)

        # Process each row (starting row number 2 for data rows)
        for row_number, row in enumerate(reader, start=2):
            caption = row["caption"]
            payload = {
                "query": caption,
                "n": 60,
                "model": args.baseline
            }
            headers = {
                "accept": "application/json",
                "Content-Type": "application/json"
            }
            try:
                # Send POST request with JSON payload
                response = requests.post(endpoint, headers=headers, json=payload)
                response.raise_for_status()  # Raise error for non-2xx responses
                data = response.json()
                # Optionally include the original caption in the result for traceability
                data["caption"] = caption
                results.append(data)
                print(f"Row {row_number} processed successfully.")
                consecutive_errors = 0
            except Exception as e:
                consecutive_errors += 1
                print(f"Error at row {row_number} (caption: {caption}): {e}", file=sys.stderr)

            # Stop if 5 consecutive errors occur
            if consecutive_errors >= 5:
                print("5 consecutive errors encountered, stopping processing.", file=sys.stderr)
                break

    # Write all successfully processed results as a JSON list to the output file
    with open(args.output_file, mode="w", encoding="utf-8") as outfile:
        json.dump(results, outfile, indent=4)


if __name__ == "__main__":
    main()
