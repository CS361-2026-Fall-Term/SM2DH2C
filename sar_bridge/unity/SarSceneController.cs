using System;
using System.IO;
using UnityEngine;

/// Renders the backend's state. Holds NO simulation logic: the drone only moves
/// because a "state" message said so. Operator input goes back as commands.
///
/// Scene setup (see README): an empty GameObject with BridgeClient + this script.
/// Assign: drone (any Transform), pathLine (LineRenderer), heatmapQuad (Renderer
/// with a transparent material), detectionMarker (any GameObject, inactive).
/// Copy terrain.raw / terrain.json from the backend's terrain_out/ into
/// Assets/StreamingAssets/ .
public class SarSceneController : MonoBehaviour
{
    [Serializable] class Envelope { public string type; }
    [Serializable] class TerrainMeta { public int n; public float size_m, cell_m, h_min, h_max; }
    [Serializable] class Hello { public TerrainMeta terrain; public float state_hz; public int grid; }
    [Serializable] class State { public float t; public bool paused; public float[] pos, quat; public int remaining; }
    [Serializable] class PathMsg { public float[] pts; }
    [Serializable] class ProbMsg { public int n; public float[] data; }
    [Serializable] class Detection { public float[] pos; public float conf; }
    [Serializable] class Cmd { public string type = "cmd"; public string cmd; public float[] pos; }

    public BridgeClient client;
    public Transform drone;
    public LineRenderer pathLine;
    public Renderer heatmapQuad;
    public GameObject detectionMarker;
    public Camera cam;
    public float followSmoothing = 4f;
    public bool chaseCam = false;

    Terrain terrain;
    TerrainMeta meta;
    Texture2D heat;
    Vector3 targetPos;
    Quaternion targetRot = Quaternion.identity;
    bool haveState;

    void Awake()
    {
        if (client == null) client = GetComponent<BridgeClient>();
        if (cam == null) cam = Camera.main;
        client.OnLine += HandleLine;
    }

    void HandleLine(string line)
    {
        switch (JsonUtility.FromJson<Envelope>(line).type)
        {
            case "hello": BuildTerrain(JsonUtility.FromJson<Hello>(line).terrain); break;
            case "state": OnState(JsonUtility.FromJson<State>(line)); break;
            case "path": OnPath(JsonUtility.FromJson<PathMsg>(line)); break;
            case "prob": OnProb(JsonUtility.FromJson<ProbMsg>(line)); break;
            case "detection": OnDetection(JsonUtility.FromJson<Detection>(line)); break;
        }
    }

    // ---- terrain --------------------------------------------------------
    void BuildTerrain(TerrainMeta m)
    {
        meta = m;
        if (terrain != null) Destroy(terrain.gameObject);

        // terrain.raw: float32 meters, row = north, col = east, both increasing.
        // Unity's SetHeights is [z(north), x(east)] in 0..1 of terrain height: same order.
        string rawPath = Path.Combine(Application.streamingAssetsPath, "terrain.raw");
        byte[] bytes = File.ReadAllBytes(rawPath);
        int n = m.n;
        if (bytes.Length != n * n * 4)
        {
            Debug.LogError($"terrain.raw is {bytes.Length} bytes, expected {n * n * 4}. Copy the current file from the backend.");
            return;
        }
        float range = Mathf.Max(m.h_max - m.h_min, 0.01f);
        var heights = new float[n, n];
        for (int r = 0; r < n; r++)
            for (int c = 0; c < n; c++)
                heights[r, c] = (BitConverter.ToSingle(bytes, (r * n + c) * 4) - m.h_min) / range;

        var td = new TerrainData { heightmapResolution = n };
        td.size = new Vector3(m.size_m, range, m.size_m);
        td.SetHeights(0, 0, heights);
        var go = Terrain.CreateTerrainGameObject(td);
        go.name = "SAR Terrain";
        // Terrain origin is its SW corner. Shift down by h_min so world Y = elevation.
        go.transform.position = new Vector3(0, m.h_min, 0);
        terrain = go.GetComponent<Terrain>();

        if (heatmapQuad != null)
        {
            // Overlay quad floats at mid elevation; swap for a projector/decal later.
            var t = heatmapQuad.transform;
            t.position = new Vector3(m.size_m / 2, m.h_max + 1f, m.size_m / 2);
            t.rotation = Quaternion.Euler(90, 0, 0);
            t.localScale = new Vector3(m.size_m, m.size_m, 1);
        }
        if (cam != null && !chaseCam)
            cam.transform.SetPositionAndRotation(
                new Vector3(m.size_m / 2, m.h_max + m.size_m * 0.8f, -m.size_m * 0.2f),
                Quaternion.Euler(70, 0, 0));
    }

    // ---- state ----------------------------------------------------------
    void OnState(State s)
    {
        targetPos = CoordConv.ToUnity(s.pos);
        targetRot = CoordConv.ToUnity(s.quat[0], s.quat[1], s.quat[2], s.quat[3]);
        if (!haveState && drone != null) { drone.SetPositionAndRotation(targetPos, targetRot); }
        haveState = true;
    }

    void Update()
    {
        if (!haveState || drone == null) return;
        // Interpolate toward the latest backend pose; Unity never simulates the drone.
        float k = 1f - Mathf.Exp(-followSmoothing * 5f * Time.deltaTime);
        drone.position = Vector3.Lerp(drone.position, targetPos, k);
        drone.rotation = Quaternion.Slerp(drone.rotation, targetRot, k);
        if (chaseCam && cam != null)
        {
            var want = drone.position + new Vector3(0, 40, -60);
            cam.transform.position = Vector3.Lerp(cam.transform.position, want, 1f - Mathf.Exp(-2f * Time.deltaTime));
            cam.transform.LookAt(drone);
        }

        HandleOperatorInput();
    }

    void OnPath(PathMsg p)
    {
        if (pathLine == null || p.pts == null) return;
        int count = p.pts.Length / 3;
        pathLine.positionCount = count;
        for (int i = 0; i < count; i++)
            pathLine.SetPosition(i, CoordConv.ToUnity(p.pts[i * 3], p.pts[i * 3 + 1], p.pts[i * 3 + 2]));
    }

    void OnProb(ProbMsg p)
    {
        if (heatmapQuad == null) return;
        if (heat == null || heat.width != p.n)
        {
            heat = new Texture2D(p.n, p.n, TextureFormat.RGBA32, false)
            { filterMode = FilterMode.Bilinear, wrapMode = TextureWrapMode.Clamp };
            heatmapQuad.material.mainTexture = heat;
        }
        float max = 1e-9f;
        foreach (var v in p.data) max = Mathf.Max(max, v);
        var px = new Color[p.n * p.n];
        // data is row-major, row = north: matches Texture2D (row 0 = bottom), and the
        // quad's rotation puts texture +v toward world +z (north).
        for (int i = 0; i < px.Length; i++)
        {
            float t = p.data[i] / max;
            px[i] = new Color(1f, 1f - t, 0f, Mathf.Clamp01(t * 0.7f)); // yellow -> red, clear when low
        }
        heat.SetPixels(px);
        heat.Apply();
    }

    void OnDetection(Detection d)
    {
        if (detectionMarker == null) return;
        detectionMarker.transform.position = CoordConv.ToUnity(d.pos);
        detectionMarker.SetActive(true);
        Debug.Log($"Person detected (conf {d.conf:0.00}) at {detectionMarker.transform.position}");
    }

    // ---- operator guidance ----------------------------------------------
    // Left click = send drone here. Shift+click = set last-known position.
    // P = pause, R = resume. All are only requests: the backend decides.
    void HandleOperatorInput()
    {
        if (cam == null) return;
        if (Input.GetKeyDown(KeyCode.P)) Send(new Cmd { cmd = "pause" });
        if (Input.GetKeyDown(KeyCode.R)) Send(new Cmd { cmd = "resume" });
        if (!Input.GetMouseButtonDown(0) || terrain == null) return;
        // Raycast the terrain collider only, so the heatmap quad can't swallow clicks.
        var tc = terrain.GetComponent<TerrainCollider>();
        if (!tc.Raycast(cam.ScreenPointToRay(Input.mousePosition), out var hit, 10000f)) return;
        var bullet = CoordConv.ToBullet(hit.point);
        bool lkp = Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift);
        Send(new Cmd { cmd = lkp ? "set_lkp" : "goto", pos = new[] { bullet[0], bullet[1] } });
    }

    void Send(Cmd c) => client.Send(JsonUtility.ToJson(c));
}
