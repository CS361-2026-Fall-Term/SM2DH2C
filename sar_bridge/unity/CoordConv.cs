using UnityEngine;

/// PyBullet world: right-handed, X east, Y north, Z up (meters).
/// Unity world:    left-handed,  X east, Y up,    Z north.
/// Swapping Y and Z is the whole conversion, and it flips handedness, so
/// rotations need their axis negated (q' = (-x, -z, -y, w)).
/// Keep ALL conversion in this file so there is one place to debug.
public static class CoordConv
{
    public static Vector3 ToUnity(float x, float y, float z) => new Vector3(x, z, y);
    public static Vector3 ToUnity(float[] p) => new Vector3(p[0], p[2], p[1]);

    /// For commands going back to the backend. Unity (x, y, z) -> PyBullet (x, z, y).
    public static float[] ToBullet(Vector3 u) => new float[] { u.x, u.z, u.y };

    public static Quaternion ToUnity(float qx, float qy, float qz, float qw)
        => new Quaternion(-qx, -qz, -qy, qw);
}
