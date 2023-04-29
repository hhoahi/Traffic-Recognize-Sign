import axios from "axios";

const postImage = (url, formData) => {
  return axios.post(url, formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });
};

const Request = { postImage };
export default Request;
