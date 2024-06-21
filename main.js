document.getElementById('upload-form').addEventListener('submit', async function(event) {
    event.preventDefault();
    const formData = new FormData();
    formData.append('image', document.getElementById('image').files[0]);
  
    const response = await fetch('/detect', {
      method: 'POST',
      body: formData
    });
  
    const result = await response.json();
    document.getElementById('results').innerText = JSON.stringify(result, null, 2);
  });
  