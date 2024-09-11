import base64
from io import BytesIO
    
# Convert image to base64 for HTML rendering
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


# Generate HTML table
def generate_html_table(cell_count):
    html = """
    <table>
        <thead>
            <tr>
                <th>Cell Type</th>
                <th>Count</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for cell_type, count in cell_count.items():
        html += f"<tr><td>{cell_type}</td><td>{count}</td></tr>"
    
    html += """
        </tbody>
    </table>
    """
    
    return html