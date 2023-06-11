function predict() {
  var fileInput = document.getElementById("image-upload");
  var file = fileInput.files[0];
  var formData = new FormData();
  formData.append("image", file);

  $.ajax({
    url: "/predict",
    type: "POST",
    data: formData,
    processData: false,
    contentType: false,
    success: function(response) {
      displayResult(response);
    },
    error: function(xhr, status, error) {
      console.log("Error:", error);
    }
  });
}

function displayResult(response) {
  var resultDiv = document.getElementById("result");
  resultDiv.innerHTML = response;
}
