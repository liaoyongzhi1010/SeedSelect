import numpy as np


def look_at(eye, target=(0, 0, 0), up=(0, 1, 0)):
    eye = np.array(eye, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    up = np.array(up, dtype=np.float32)
    z = eye - target
    z = z / (np.linalg.norm(z) + 1e-8)
    x = np.cross(up, z)
    x = x / (np.linalg.norm(x) + 1e-8)
    y = np.cross(z, x)
    mat = np.eye(4, dtype=np.float32)
    mat[:3, 0] = x
    mat[:3, 1] = y
    mat[:3, 2] = z
    mat[:3, 3] = eye
    return mat


def orbit_cameras(radius=2.0, elevations=(30, 0, -30), azimuths=(0, 45, 90, 135, 180, 225, 270, 315)):
    cams = []
    for elev_deg in elevations:
        elev = np.deg2rad(elev_deg)
        for az_deg in azimuths:
            az = np.deg2rad(az_deg)
            x = radius * np.cos(elev) * np.sin(az)
            y = radius * np.sin(elev)
            z = radius * np.cos(elev) * np.cos(az)
            mat = look_at((x, y, z))
            cams.append({
                'elev': elev_deg,
                'azim': az_deg,
                'c2w': mat,
            })
    return cams


def canonical_zero123pp_cameras(radius=2.0):
    elevations = [20, -10, 20, -10, 20, -10]
    azimuths = [30, 90, 150, 210, 270, 330]
    cams = []
    for elev_deg, az_deg in zip(elevations, azimuths):
        elev = np.deg2rad(elev_deg)
        az = np.deg2rad(az_deg)
        x = radius * np.cos(elev) * np.sin(az)
        y = radius * np.sin(elev)
        z = radius * np.cos(elev) * np.cos(az)
        mat = look_at((x, y, z))
        cams.append({
            'elev': elev_deg,
            'azim': az_deg,
            'c2w': mat,
        })
    return cams
