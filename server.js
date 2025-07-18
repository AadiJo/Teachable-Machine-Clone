const express = require("express");
const path = require("path");
const { createProxyMiddleware } = require("http-proxy-middleware");

const app = express();
const port = 3000;

// Serve static files
app.use(express.static(path.join(__dirname)));

// Setup proxy for API requests to Python backend
app.use(
  "/api",
  createProxyMiddleware({
    target: "http://localhost:5000", // Python server address
    changeOrigin: true,
    pathRewrite: {
      "^/api": "/api", // maintain the /api path when forwarding
    },
    onError: (err, req, res) => {
      console.error("Proxy error:", err);
      res
        .status(500)
        .json({
          error:
            'Python backend connection failed. Make sure to run "python app.py" first.',
        });
    },
  })
);

// Serve main HTML file
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "index.html"));
});

app.listen(port, () => {
  console.log(`Frontend server running at http://localhost:${port}`);
  console.log(`Remember to start the Python backend: python app.py`);
});
