using System;
using System.Collections.Concurrent;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

/// TCP client for the PyBullet backend (newline-delimited JSON).
/// A background thread reads lines; Update() on the main thread drains them,
/// because Unity APIs may only be touched from the main thread.
public class BridgeClient : MonoBehaviour
{
    public string host = "127.0.0.1";
    public int port = 5555;
    public float reconnectSeconds = 2f;

    public event Action<string> OnLine;       // raw JSON line, main thread
    public event Action OnConnected, OnDisconnected;
    public bool IsConnected { get; private set; }

    readonly ConcurrentQueue<string> inbox = new ConcurrentQueue<string>();
    readonly ConcurrentQueue<string> outbox = new ConcurrentQueue<string>();
    Thread worker;
    volatile bool running;
    volatile bool connectedFlag, disconnectedFlag;

    void OnEnable()
    {
        running = true;
        worker = new Thread(Loop) { IsBackground = true };
        worker.Start();
    }

    void OnDisable()
    {
        running = false;
    }

    public void Send(string json) => outbox.Enqueue(json);

    void Loop()
    {
        while (running)
        {
            try
            {
                using (var tcp = new TcpClient { NoDelay = true })
                {
                    tcp.Connect(host, port);
                    connectedFlag = true;
                    var stream = tcp.GetStream();
                    var chunk = new byte[8192];
                    var pending = new System.Collections.Generic.List<byte>(16384);
                    while (running && tcp.Connected)
                    {
                        while (outbox.TryDequeue(out var o))
                        {
                            var bytes = new UTF8Encoding(false).GetBytes(o + "\n");
                            stream.Write(bytes, 0, bytes.Length);
                        }
                        if (tcp.Available == 0)
                        {
                            // Poll() also detects a closed peer (readable with 0 bytes).
                            if (tcp.Client.Poll(20000, SelectMode.SelectRead) && tcp.Available == 0) break;
                            continue;
                        }
                        int n = stream.Read(chunk, 0, chunk.Length);
                        if (n <= 0) break;
                        for (int i = 0; i < n; i++)
                        {
                            if (chunk[i] == (byte)'\n')
                            {
                                inbox.Enqueue(Encoding.UTF8.GetString(pending.ToArray()));
                                pending.Clear();
                            }
                            else pending.Add(chunk[i]);
                        }
                    }
                }
            }
            catch (Exception) { /* connection refused etc.: retry below */ }
            disconnectedFlag = true;
            for (float t = 0; running && t < reconnectSeconds; t += 0.1f) Thread.Sleep(100);
        }
    }

    void Update()
    {
        if (connectedFlag) { connectedFlag = false; IsConnected = true; OnConnected?.Invoke(); }
        if (disconnectedFlag) { disconnectedFlag = false; IsConnected = false; OnDisconnected?.Invoke(); }
        int budget = 200; // cap per frame so a burst can't stall rendering
        while (budget-- > 0 && inbox.TryDequeue(out var line)) OnLine?.Invoke(line);
    }
}
