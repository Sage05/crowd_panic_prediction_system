import { useState, useEffect, useRef } from "react";

export function useLiveSocket(url = "ws://localhost:8000/ws/live") {
  const [state, setState] = useState(null);
  const ws = useRef(null);

  useEffect(() => {
    let retry;
    const connect = () => {
      ws.current = new WebSocket(url);
      ws.current.onmessage = (e) => {
  const data = JSON.parse(e.data);
  data.timestamp = new Date(data.timestamp);
  console.log("WS DATA:",data); // ← add this line
  setState(data);
};
      ws.current.onclose = () => { retry = setTimeout(connect, 2000); }; // auto-reconnect
      ws.current.onerror = () => ws.current.close();
    };
    connect();
    return () => { clearTimeout(retry); ws.current?.close(); };
  }, [url]);

  return state;
}