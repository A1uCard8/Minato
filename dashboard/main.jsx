import { useEffect, useState } from "react";

function App() {
  const [metrics, setMetrics] = useState(null);

  const fetchMetrics = async () => {
    const res = await fetch("http://localhost:8000/benchmark");
    const data = await res.json();
    setMetrics(data);
  };

  useEffect(() => {
    fetchMetrics();
  }, []);

  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4">Edge AI Model Benchmark</h1>
      {metrics ? (
        <div>
          <p><strong>Model:</strong> {metrics.model_name}</p>
          <p><strong>Latency:</strong> {metrics.latency_ms.toFixed(2)} ms</p>
          <p><strong>Memory Usage:</strong> {metrics.memory_mb.toFixed(2)} MB</p>
          <p><strong>Output Shape:</strong> {metrics.output_shape.join("x")}</p>
        </div>
      ) : (
        <p>Loading metrics...</p>
      )}
    </div>
  );
}

export default App;
