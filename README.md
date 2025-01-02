# Docling Serve

 Running [Docling](https://github.com/DS4SD/docling) as an API service.

 > [!NOTE]
> This is an unstable draft implementation which will quickly evolve.

## Development

Install the dependencies

```sh
# Install poetry if not already available
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Run the server
poetry run uvicorn docling_serve.app:app --reload
```

Example payload (http source):

```sh
curl -X 'POST' \
  'http://127.0.0.1:8000/convert' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "http_source": {
    "url": "https://arxiv.org/pdf/2206.01062"
  }
}'
```

Example extracting Markdown with a placeholder for images (http source):

```sh
curl -s -X POST "http://localhost:8000/convert/markdown" -H "Content-Type: application/json" -d '{
  "options": {
    "include_images": false
  },
  "http_source": {
    "url": "https://arxiv.org/pdf/2206.01062"
  }
}' > output.md
```

Example posting a file for conversion or explicit Markdown conversion:

When your PDF or other file type is too large, encoding it as a base64 string
and passing it inline to curl can lead to an “Argument list too long” error on
some systems. To avoid this, we write the JSON request body to a file and have
curl read from that file.

```sh
# 1. Base64-encode the file
B64_DATA=$(base64 -w 0 /path/to/file/pdf-to-convert.pdf)

# 2. Build the JSON with your options
cat <<EOF > /tmp/request_body.json
{
  "options": {
    "output_markdown": true,
    "include_images": false
  },
  "file_source": {
    "base64_string": "${B64_DATA}",
    "filename": "pdf-to-convert.pdf"
  }
}
EOF

# 3. POST the request to the docling service
curl -X POST "http://localhost:8000/convert" \
     -H "Content-Type: application/json" \
     -d @/tmp/request_body.json

# Or explicitly convert to Markdown
curl -X POST "http://localhost:8000/convert/markdown" \
     -H "Content-Type: application/json" \
     -d @/tmp/request_body.json
```

### Cuda GPU Support

For GPU support try the following:

```sh
# Create a virtual env
python3 -m venv venv

# Activate the venv
source venv/bin/active

# Install torch with the special index
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install the package
pip install -e .

# Run the server
poetry run uvicorn docling_serve.app:app --reload
```
