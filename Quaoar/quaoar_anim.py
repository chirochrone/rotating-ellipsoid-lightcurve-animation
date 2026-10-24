import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from scipy.spatial import ConvexHull

# Ellipsoid radii
a, b, c = 1241.4/2, 1093.0/2, 1023.0/2
# Geometric albedo
albedo = 0.111

# Define color map name (string) and opacity of ellipsoid model
colormap = "copper"
ellipsoid_opacity = 1   # keep at 1; translucent contourf causes rendering issues :(

# Tilt the viewing perspective
'''
Opening angle of 18.1 deg from Proudfoot et al. (2025) - https://iopscience.iop.org/article/10.3847/PSJ/addd02/
Position angle of 350.2 deg from Pereira et al. (2023) - https://www.aanda.org/articles/aa/pdf/2023/05/aa46365-23.pdf
'''
view_pitch = np.deg2rad(90+18.1)   # tilt toward/away. Equivalent to 90 - (opening angle).
view_yaw = np.deg2rad(350-180)    # rotate clockwise (negative) or counterclockwise (positive)

# Sampling grid
u = np.linspace(0, 2 * np.pi, 80)
v = np.linspace(0, np.pi, 40)
U, V = np.meshgrid(u, v)

ROTATION_PERIOD = 17.6792  # [hours] this parameter only changes plot labels
FRAMES_PER_ROTATION = 60
ANGLE_PER_FRAME = 2 * np.pi / FRAMES_PER_ROTATION

# Parametric ellipsoid in body coordinates
X = a * np.cos(U) * np.sin(V)
Y = b * np.sin(U) * np.sin(V)
Z = c * np.cos(V)

# Rotation about the fixed shortest axis (Z)
def rotate_points(x, y, z, angle):
    cz, sz = np.cos(angle), np.sin(angle)
    x_rot = cz * x - sz * y
    y_rot = sz * x + cz * y
    return x_rot, y_rot, z

def rotate_normals(nx, ny, nz, angle):
    """Rotate surface normals (same as rotate_points)"""
    cz, sz = np.cos(angle), np.sin(angle)
    nx_rot = cz * nx - sz * ny
    ny_rot = sz * nx + cz * ny
    return nx_rot, ny_rot, nz

def apply_view_transform(x, y, z):
    cp, sp = np.cos(view_pitch), np.sin(view_pitch)
    cy, sy = np.cos(view_yaw), np.sin(view_yaw)

    y1 = cp * y - sp * z
    z1 = sp * y + cp * z

    x1 = cy * x - sy * y1
    y2 = sy * x + cy * y1
    return x1, y2, z1

def apply_view_transform_normals(nx, ny, nz):
    """Apply view transform to normals (same transformation)"""
    cp, sp = np.cos(view_pitch), np.sin(view_pitch)
    cy, sy = np.cos(view_yaw), np.sin(view_yaw)

    ny1 = cp * ny - sp * nz
    nz1 = sp * ny + cp * nz

    nx1 = cy * nx - sy * ny1
    ny2 = sy * nx + cy * ny1
    return nx1, ny2, nz1

# Create figure with two subplots
gs = GridSpec(2, 1, height_ratios=[2, 1], hspace=0.3)
fig = plt.figure(figsize=(6, 8), facecolor="black")
fig.subplots_adjust(left=0.04, right=0.96, top=0.93, bottom=0.1, hspace=0.42)
ax = fig.add_subplot(gs[0])
ax_H = fig.add_subplot(gs[1])

# Define plot font sizes
fs_axis = 16
fs_ticks = 14
fs_title = 16

# Use a black figure and plotting areas.
fig.patch.set_facecolor("black")
ax.set_facecolor("black")
ax.set_title("Quaoar Viewed From Earth", color="white", fontsize=fs_title)
ax_H.set_facecolor("black")

ax.set_aspect("equal")
bordersize = 1.5 * a
ax.set_xlim(-bordersize, bordersize)
ax.set_ylim(-bordersize, bordersize)
ax.set_xticks([])
ax.set_yticks([])

contour = ax.contourf(*apply_view_transform(X, Y, Z), levels=30, cmap=colormap, alpha=ellipsoid_opacity)
mesh_lines = []

# Match the lower panel's border width to the upper panel's actual width.
# The upper axes is narrower than its GridSpec cell because its aspect is equal.
fig.canvas.draw()
upper_pos = ax.get_position()
lower_pos = ax_H.get_position()
ax_H.set_position([
    upper_pos.x0,
    lower_pos.y0,
    upper_pos.width,
    lower_pos.height,
])

# Mesh grid parameters.
meridian_angles = np.linspace(0, 2 * np.pi, 20, endpoint=False)  # longitude
parallel_angles = np.linspace(0.2, np.pi - 0.2, 12)  # latitude

def style_H_axis():
    # Apply the black-background, white-text style to the lightcurve axes.
    ax_H.set_facecolor("black")
    ax_H.set_xlabel("Time (hours)", color="white", fontsize=fs_axis)
    ax_H.set_ylabel("Relative Mag.", color="white", fontsize=fs_axis)
    ax_H.set_title("Rotational Lightcurve\n(Magnitude Over Time)", color="white", fontsize=fs_title)
    ax_H.tick_params(axis="both", colors="white", labelsize=fs_ticks)
    for spine in ax_H.spines.values():
        spine.set_color("white")
    ax_H.grid(True, color="white", ls=':', alpha=0.25, zorder=-9)

def compute_projection_area(angle):
    """Compute the area of the 2D projection of the visible ellipsoid"""
    x_rot, y_rot, z_rot = rotate_points(X, Y, Z, angle)
    x_view, y_view, z_view = apply_view_transform(x_rot, y_rot, z_rot)
    
    # Get all visible points (front-facing)
    visible_mask = z_view >= 0
    if not np.any(visible_mask):
        return 0.0
    
    # Get visible points
    x_visible = x_view[visible_mask].flatten()
    y_visible = y_view[visible_mask].flatten()
    
    # Find the convex hull to get the outline
    if len(x_visible) < 3:
        return 0.0
    
    points = np.column_stack((x_visible, y_visible))
    try:
        hull = ConvexHull(points)
        # Compute area using shoelace formula
        hull_points = points[hull.vertices]
        area = 0.5 * np.abs(np.dot(hull_points[:, 0], np.roll(hull_points[:, 1], 1)) - 
                           np.dot(hull_points[:, 1], np.roll(hull_points[:, 0], 1)))
        return area
    except:
        return 0.0

def abs_mag(area, albedo):
    D = np.sqrt(area / np.pi)  # circle diameter in km
    H = -5*np.log10(D * np.sqrt(albedo) / 1329)
    return H

def draw_mesh(angle):
    lines = []
    for u0 in meridian_angles:
        xm = a * np.cos(u0) * np.sin(v)
        ym = b * np.sin(u0) * np.sin(v)
        zm = c * np.cos(v)
        # Compute surface normals (proportional to (x/a², y/b², z/c²))
        nx = xm / (a * a)
        ny = ym / (b * b)
        nz = zm / (c * c)
        # Normalize
        norm = np.sqrt(nx*nx + ny*ny + nz*nz)
        nx, ny, nz = nx/norm, ny/norm, nz/norm
        
        xr, yr, zr = rotate_points(xm, ym, zm, angle)
        nxr, nyr, nzr = rotate_normals(nx, ny, nz, angle)
        xv, yv, zv = apply_view_transform(xr, yr, zr)
        nxv, nyv, nzv = apply_view_transform_normals(nxr, nyr, nzr)
        
        # Front-facing if normal z-component points toward viewer (positive)
        mask = nzv > 0
        if np.any(mask):
            diff_mask = np.diff(np.concatenate(([False], mask, [False])).astype(int))
            starts = np.where(diff_mask == 1)[0]
            ends = np.where(diff_mask == -1)[0]
            for start, end in zip(starts, ends):
                lines.append(ax.plot(xv[start:end], yv[start:end], color="black", lw=0.5, alpha=0.75)[0])
    for v0 in parallel_angles:
        xp = a * np.cos(u) * np.sin(v0)
        yp = b * np.sin(u) * np.sin(v0)
        zp = np.full_like(u, c * np.cos(v0))
        # Compute surface normals
        nx = xp / (a * a)
        ny = yp / (b * b)
        nz = zp / (c * c)
        norm = np.sqrt(nx*nx + ny*ny + nz*nz)
        nx, ny, nz = nx/norm, ny/norm, nz/norm
        
        xr, yr, zr = rotate_points(xp, yp, zp, angle)
        nxr, nyr, nzr = rotate_normals(nx, ny, nz, angle)
        xv, yv, zv = apply_view_transform(xr, yr, zr)
        nxv, nyv, nzv = apply_view_transform_normals(nxr, nyr, nzr)
        
        mask = nzv > 0
        if np.any(mask):
            diff_mask = np.diff(np.concatenate(([False], mask, [False])).astype(int))
            starts = np.where(diff_mask == 1)[0]
            ends = np.where(diff_mask == -1)[0]
            for start, end in zip(starts, ends):
                lines.append(ax.plot(xv[start:end], yv[start:end], color="black", lw=0.5, alpha=0.75)[0])
    return lines

# ---------------------------------------------------------
# Precompute full-rotation min/max area (or magnitude)
# ---------------------------------------------------------

# Compute projected area for all frames of ONE rotation
areas_full = np.array([compute_projection_area(i * ANGLE_PER_FRAME)
                       for i in range(FRAMES_PER_ROTATION)])

# Convert to magnitudes
H_full = abs_mag(areas_full, albedo)

# Compute global limits
H_min, H_max = np.min(H_full), np.max(H_full)

# Take the mean of the min and max
H_mean = (H_min + H_max) / 2

# Add padding
H_range = H_max - H_min
padding = 1.05 * 0.3#H_range if H_range > 0 else 1.05

H_ymin = -padding  # center the plot around y=0
H_ymax = padding

# Storage for H over time
H_history = []
time_history = []
H_line = None

t = np.linspace(0, ROTATION_PERIOD, 200)
def lightcurve_preview():
    # Plot a preview of the ellipsoid lightcurve, which is just a cosine wave with semi-amplitude = -H_range/2
    preview_lightcurve = (-H_range/2)*np.cos(4*np.pi*t/ROTATION_PERIOD)
    ax_H.plot(t, preview_lightcurve, ls='-', lw=1.5, color='gold', alpha=0.25, zorder=-2)

    # Add a double-headed arrow to indicate the peak-to-trough amplitude
    annotation_padding = 1.05   # 0.08 is the distance between the annotation and right border of lightcurve plot
    ax_H.annotate(
        "",#f"Δm = {H_range:.2f} mag",   # the text offsets the arrow, abandon this and use a separate text annotation
        xy=(annotation_padding*ROTATION_PERIOD, 1.1*H_range/2),  # End point (head 1)
        xytext=(annotation_padding*ROTATION_PERIOD, -1.1*H_range/2),  # Start point (head 2)
        ha='center',    # Centers text horizontally between xy and xytext
        va='center',    # Sits text slightly above the arrow line
        fontsize=fs_ticks,    # Adjust text size independently
        color='white',
        arrowprops=dict(arrowstyle="<->", color="white", lw=2, mutation_scale=15),  # default arrowhead mutation scale is 10
        annotation_clip=False  # prevents clipping outside the axis frame, so the arrow can show
    )
    ax_H.text((annotation_padding+0.1)*ROTATION_PERIOD, 0, f"Δm\n{H_range:.2f}", fontsize=fs_ticks, color="white", ha="center", va="center")

def init():
    global contour, mesh_lines, H_history, time_history, H_line
    H_history = []
    time_history = []
    # Clear existing collections safely
    if hasattr(contour, 'collections'):
        for coll in list(contour.collections):
            try:
                coll.remove()
            except:
                pass
    contour = ax.contourf(*apply_view_transform(X, Y, Z), levels=60, cmap=colormap, alpha=ellipsoid_opacity)
    mesh_lines = draw_mesh(0.0)
    
    # Initialize H plot
    ax_H.clear()
    style_H_axis()
    lightcurve_preview()
    ax_H.set_xlim(0, ROTATION_PERIOD)
    H_line, = ax_H.plot([], [], color="yellow", lw=2)

    initial_area = compute_projection_area(0.0)
    initial_H = abs_mag(initial_area, albedo)
    H_history.append(initial_H)
    time_history.append(0.0)
    H_line.set_data(time_history, H_history)

    if H_history:
        ax_H.set_ylim(H_ymin, H_ymax)
        ax_H.invert_yaxis()

    return contour.collections + mesh_lines + [H_line]

def update(frame):
    global contour, mesh_lines, H_history, time_history, H_line
    angle = frame * ANGLE_PER_FRAME
    cycle_frame = frame % FRAMES_PER_ROTATION

    if hasattr(contour, 'collections'):
        for coll in list(contour.collections):
            try:
                coll.remove()
            except:
                pass

    for line in list(mesh_lines):
        try:
            line.remove()
        except:
            pass
    mesh_lines = []

    x_rot, y_rot, z_rot = rotate_points(X, Y, Z, angle)
    contour = ax.contourf(*apply_view_transform(x_rot, y_rot, z_rot), levels=30, cmap=colormap, alpha=ellipsoid_opacity)
    mesh_lines = draw_mesh(angle)

    if cycle_frame == 0 and H_history:
        H_history.clear()
        time_history.clear()
        H_line.set_data([], [])

    projection_area = compute_projection_area(angle)
    H_history.append(abs_mag(projection_area, albedo)-H_mean)
    time_history.append(cycle_frame * ROTATION_PERIOD / FRAMES_PER_ROTATION)
    H_line.set_data(time_history, H_history)

    if H_history:
        ax_H.set_ylim(H_ymin, H_ymax)
        ax_H.invert_yaxis()
    ax_H.set_xlim(0, ROTATION_PERIOD)

    return contour.collections + mesh_lines + [H_line]

# Ensure any visible frame on the upper panel is white.
ax.tick_params(axis="both", colors="white")
for spine in ax.spines.values():
    spine.set_color("white")

# Save the animation as a gif
ani = FuncAnimation(fig, update, init_func=init, frames=FRAMES_PER_ROTATION, interval=30, blit=False)
ani.save('quaoar_rotation.gif', writer='pillow', fps=60)  # save animation as gif
plt.show()