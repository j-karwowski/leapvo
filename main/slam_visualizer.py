import os
import json

import cv2
import matplotlib as mpl
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from moviepy.editor import ImageSequenceClip


def read_video_from_path(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Error opening video file")
    else:
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                frames.append(np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            else:
                break
        cap.release()
    return np.stack(frames)


class SLAMVisualizer:
    def __init__(
        self,
        cfg,
        save_dir=None,
        # grayscale: bool = False,
        # pad_value: int = 0,
        # fps: int = 10,
        # mode: str = "rainbow",  # 'cool', 'optical_flow'
        # linewidth: int = 2,
        # show_first_frame: int = 10,
        # tracks_leave_trace: int = 0,  # -1 for infinite
    ):
        self.cfg = cfg.visualizer
        self.cfg_full = cfg
        self.mode = self.cfg.mode
        self.save_dir = self.cfg.save_dir
        if save_dir is not None:
            self.save_dir = save_dir

        # print(f'cfg full: {self.cfg_full}')

        if self.cfg.mode == "rainbow":
            self.color_map = mpl.colormaps["gist_rainbow"]
        elif self.cfg.mode == "cool":
            self.color_map = mpl.colormaps[self.cfg.mode]
        self.show_first_frame = self.cfg.show_first_frame
        self.grayscale = self.cfg.grayscale
        self.tracks_leave_trace = self.cfg.tracks_leave_trace
        self.pad_value = self.cfg.pad_value
        self.linewidth = self.cfg.linewidth
        self.fps = self.cfg.fps

        # storage

        self.frames = []
        self.tracks = []

    def add_frame(self, frame):
        self.frames.append(frame)

    def add_track(self, track):
        self.tracks.append(track)

    # TODO: This method exists double and is never called
    def draw_tracks_on_frames(self):
        # print('In draw_tracks_on_frames() (in line 74):')
        video = torch.stack(self.frames, dim=0)
        video = F.pad(
            video,
            (self.pad_value, self.pad_value, self.pad_value, self.pad_value),
            "constant",
            255,
        )
        video = video.permute(0, 2, 3, 1).detach().cpu().numpy()

        res_video_sta = []
        res_video_dyn = []
        # process input video
        for rgb in video:
            # print(f'- rgb: {rgb}')
            res_video_sta.append(rgb.copy())
            res_video_dyn.append(rgb.copy())

        T = self.fps * 2  # period of color repetition

        for t, track in enumerate(self.tracks):
            # print(f'- For loop over T {t} - Track: {track}')

            targets = track["targets"][0].long().detach().cpu().numpy()
            targets = targets + self.pad_value
            S, N, _ = targets.shape

            # print(f'-- Targets ({targets.shape}) - ex. {targets[0][0]}')

            vis_label = None
            static_label = None
            coords_vars = None

            if "vis_label" in track:
                vis_label = track["vis_label"][0].detach().cpu().numpy()
                # print(f'-- vis_label: {vis_label}')
            if "static_label" in track:
                static_label = track["static_label"][0].detach().cpu().numpy()
                # print(f'-- static_label: {static_label}')
            if "coords_vars" in track:
                coords_vars = track["coords_vars"][0].detach().cpu().numpy()
                # print(f'-- Coords_vars: {coords_vars}')

            for s in range(S):
                color = (
                    np.array(self.color_map(((t - S + 1 + s) % T) / T)[:3])[None] * 255
                )
                vector_colors = np.repeat(color, N, axis=0)

                # print(f'-- color: {color}')
                # print(f'-- vector_colors: {vector_colors}')

                for n in range(N):
                    coord = (targets[s, n, 0], targets[s, n, 1])
                    # print(f'-- for loop n {n} - coord: {coord}')
                    visibile = True
                    if vis_label is not None:
                        visibile = vis_label[s, n]
                    static = True
                    if static_label is not None:
                        static = static_label[s, n]
                    if coords_vars is not None:
                        # conf_scale = np.sqrt(coords_vars[s,n]) * 3
                        conf_scale = 4 - 3 * np.exp(-coords_vars[s, n])
                    else:
                        conf_scale = 1.0

                    if coord[0] != 0 and coord[1] != 0:

                        radius = int(self.linewidth * 2)
                        if static:
                            cv2.circle(
                                res_video_sta[t],
                                coord,
                                radius,
                                vector_colors[n].tolist(),
                                thickness=-1 if visibile else 2 - 1,
                            )
                            cv2.circle(
                                res_video_sta[t],
                                coord,
                                int(radius * conf_scale * 3),
                                vector_colors[n].tolist(),
                                2 - 1,
                            )
                        else:
                            cv2.circle(
                                res_video_dyn[t],
                                coord,
                                radius,
                                vector_colors[n].tolist(),
                                thickness=-1 if visibile else 2 - 1,
                            )
        #  construct the final rgb sequence

        res_video = []
        for i in range(len(video)):
            frame_combine = np.concatenate([res_video_sta[i], res_video_dyn[i]], axis=0)
            res_video.append(frame_combine)

        #  construct the final rgb sequence
        if self.show_first_frame > 0:
            res_video = [res_video[0]] * self.show_first_frame + res_video[1:]
        return torch.from_numpy(np.stack(res_video)).permute(0, 3, 1, 2)[None].byte()

    def save_video(self, filename, writer=None, step=0):
        video = self.draw_tracks_on_frames()

        # export video
        if writer is not None:
            writer.add_video(
                f"{filename}_pred_track",
                video.to(torch.uint8),
                global_step=step,
                fps=self.fps,
            )
        else:
            os.makedirs(self.save_dir, exist_ok=True)
            wide_list = list(video.unbind(1))
            wide_list = [wide[0].permute(1, 2, 0).cpu().numpy() for wide in wide_list]
            clip = ImageSequenceClip(wide_list[2:-1], fps=self.fps)

            # Write the video file
            save_path = os.path.join(self.save_dir, f"{filename}_pred_track.mp4")
            clip.write_videofile(save_path, codec="libx264", fps=self.fps, logger=None)

            print(f"Video saved to {save_path}")


class LEAPVisualizer(SLAMVisualizer):
    def __init__(
        self,
        cfg,
        save_dir=None,
    ):
        super(LEAPVisualizer, self).__init__(cfg=cfg, save_dir=save_dir)

    def add_frame(self, frame):
        self.frames.append(frame)

    def add_track(self, track):
        self.tracks.append(track)

    def _draw_pred_tracks(
        self,
        rgb: np.ndarray,  # H x W x 3
        tracks: np.ndarray,  # T x 2
        vector_colors: np.ndarray,
        alpha: float = 0.5,
    ):
        # print('-- In _draw_pred_tracks().')
        T, N, _ = tracks.shape

        # print(f'--- Tracks ({tracks.shape})')

        for s in range(T - 1):
            vector_color = vector_colors[s]
            original = rgb.copy()
            alpha = (s / T) ** 2
            for i in range(N):
                # Again in for loop over t and n, coordinates are extracted
                coord_y = (int(tracks[s, i, 0]), int(tracks[s, i, 1]))
                coord_x = (int(tracks[s + 1, i, 0]), int(tracks[s + 1, i, 1]))
                if coord_y[0] != 0 and coord_y[1] != 0:
                    cv2.line(
                        rgb,
                        coord_y,
                        coord_x,
                        vector_color[i].tolist(),
                        self.linewidth,
                        cv2.LINE_AA,
                    )
            if self.tracks_leave_trace > 0:
                rgb = cv2.addWeighted(rgb, alpha, original, 1 - alpha, 0)
        return rgb

    def draw_tracks_on_video(
        self,
        video: torch.Tensor,
        tracks: torch.Tensor,
        vector_colors: torch.Tensor = None,
        visibility: torch.Tensor = None,
        variances: torch.Tensor = None,
    ):
        # print('- In draw_tracks_on_video()')
        B, T, C, H, W = video.shape
        _, _, N, D = tracks.shape

        # print(f'-- T {T} and N {N}')

        # print(f'-- tracks ({tracks.shape})')    
        # print(f'-- video ({video.shape})')          

        assert D == 2
        assert C == 3
        video = video[0].permute(0, 2, 3, 1).byte().detach().cpu().numpy()  # S, H, W, C
        tracks = tracks[0].long().detach().cpu().numpy()  # S, N, 2

        res_video = []

        # Minor customization of visualization method: White video background
        # white_rgb = np.full((H, W, C), 255, dtype=np.uint8) # ends up overriding the dots put in the video in previous method calls

        # process input video
        for rgb in video:
            # print(f'- rgb ({rgb.shape}) and white_rgb ({white_rgb.shape})')
            res_video.append(rgb.copy()) 
            # res_video.append(white_rgb.copy())

        # vector_colors = np.zeros((T, N, 3))

        # for t in range(T):
        #     color = np.array(self.color_map(t / T)[:3])[None] * 255
        #     vector_colors[t] = np.repeat(color, N, axis=0)

        #  draw tracks
        # print(f'-- Drawing tracks in for loop from 1 to T {T}')
        if self.tracks_leave_trace != 0:
            for t in range(1, T): # loop over frames starting at 1
                # print(f'- For loop over T iterarion {t}')
                first_ind = (
                    max(0, t - self.tracks_leave_trace) # define start of trace
                    if self.tracks_leave_trace >= 0
                    else 0
                )
                curr_tracks = tracks[first_ind : t + 1] # get past track positions
                curr_colors = vector_colors[first_ind : t + 1] # get past colors

                # print(f'--- curr_tracks ({curr_tracks.shape})') #  -- ex. {curr_tracks[0][0]}')
                # print(f'--- curr_colors ({curr_colors.shape})') #  -- ex. {curr_colors[0][0]}') 

                res_video[t] = self._draw_pred_tracks(
                    res_video[t], # current frame
                    curr_tracks, # tracks up to current time
                    curr_colors, # corrsponding colors
                )

        #  draw points
        # f'-- Drawing points with their coords in for loop over T {T} and N {N}')
        for t in range(T): # loop through all frames in T --> Temporal loop
            invalid_coords = 0
            for i in range(N): # loop through all N tracked objects

                # print(f'- For loop over T and N iteration {t} - {i}')
                coord = (tracks[t, i, 0], tracks[t, i, 1]) # extraxt track coordinates
                # print(f'-- coord: {coord}')

                visibile = True
                if visibility is not None:
                    visibile = visibility[0, t, i] # check if track is visible

                if coord[0] != 0 and coord[1] != 0: # ignore invalid coordinates
                    cv2.circle(
                        res_video[t],
                        coord,
                        int(self.linewidth * 2), # circle size
                        vector_colors[t, i].tolist(), # circle color
                        # TODO: make visibility "optional"?
                        thickness=-1 if visibile else 2 - 1, # fill if visible, otherwise outline
                    )

                    # print(f"-- Frame {t}, Track {i}, Variance: {variances[0, t, i]}, Scale: {conf_scale}")

                    # draw uncertainty
                    # TODO: investigate this methods effect on resulting visualizations
                    if variances is not None:
                        # conf_scale = np.sqrt(coords_vars[s,n]) * 3
                        conf_scale = 4 - 3 * np.exp(-variances[0, t, i]) # scale uncertainty
                        overlay = res_video[t].copy()
                        cv2.circle(
                            overlay, # img
                            coord, # circle center
                            int(self.linewidth * 2 * conf_scale * 3), # radius # larger transparent uncertainty circle
                            # mpl.colors.to_rgba("black"), # TODO: check, where/if high variance dots are visible
                            vector_colors[t, i].tolist(), # color
                            1, # thickness
                            -1, # TODO: Does this make sense here?
                        )
                        alpha = 0.5
                        res_video[t] = cv2.addWeighted( # blends overlay (with variance circles) and res_video[t] (original frame) using alpha blending
                            overlay, alpha, res_video[t], 1 - alpha, 0
                        )
                else:
                    # coord is considered invalid with (0,0)
                    # print(f"-- invalid coord: frame {t}, track {i}, coord {coord}")
                    invalid_coords += 1

            # print(f'-- in frame {t} there were: {N} tracked points and {invalid_coords} invalid tracks.')

        return torch.from_numpy(np.stack(res_video)).permute(0, 3, 1, 2)
    
    def tensor_to_list(self, obj):
        """Recursively converts all tensors in a structure (list/dict) to lists."""
        if isinstance(obj, torch.Tensor):
            return obj.tolist()  
        elif isinstance(obj, dict):
            return {k: self.tensor_to_list(v) for k, v in obj.items()}  
        elif isinstance(obj, list):
            return [self.tensor_to_list(v) for v in obj] 
        else:
            return obj  
        
    def draw_tracks_on_frames(self):
        print('In draw_tracks_on_frames() (in line 350):')
        video = torch.stack(self.frames, dim=0) # generate video by stacking individual frames
        N, C, H, W = video.shape 
        # print(f'video shape: {video.shape}')

        # customization here: Make video all white, to get clearer view of track movements
        # video = torch.full((N, C, H, W), 255, dtype=video.dtype)

        video = F.pad( # padding video equally on all sides of video (value is 0)
            video,
            (self.pad_value, self.pad_value, self.pad_value, self.pad_value),
            "constant",
            255,
        )
        # print(f'- video length in frames: {len(video)}')

        # customization here: Drawing all dots on one single video
        # res_video_sta = video.clone()
        # res_video_dyn = video.clone()
        combined_video = video.clone()

        T = self.fps  # period of color repetition (default 10)
        # print(f'- self.tracks ({len(self.tracks)})') # 25 for sintel # 50 for my data, i.e. every other frame is skipped
        
        last_frame = 0
        current_frame = 0
        current_fid = 0
        repetition = self.cfg_full.slam.S_slam/self.cfg_full.slam.kf_stride 

        for t, track in enumerate(self.tracks):
            # print(f'- For loop over T {t} - Track with FID: {track["fid"]}')
            fid = track["fid"] # frame id
            

            # Customization: Only use every 6th frame with FID 
            # if t>0 and ((t%repetition) == 0 or t == len(self.tracks)):
            #     current_fid = fid
            # else:
            #     print(f'-- Skip track with FID {fid} - due to overlap.')
            #     continue

            targets = track["targets"] + self.pad_value # coordinates of tracked points

            # current_frame += len(targets[0]) 
            # frames_to_process = current_frame - last_frame 
            frames_to_process = len(targets[0]) # number of frames to process
            # print(f'-- current_frame {current_frame} and last_frame {last_frame} - frames_to_process {frames_to_process}')    
            # last_frame = current_frame

            keyframe = 0 # TODO: Which keyframe should we choose or should we choose multiple keyframes?
            # remove every row in the 3rd dimension of targets except of the keyframe row
            zero_targets = torch.zeros_like(targets)
            zero_targets[:, -frames_to_process:, keyframe, :, :] = targets[:, -frames_to_process:, keyframe, :, :]
            custom_targets = targets[:, -frames_to_process:, keyframe:keyframe+1, :, :]
            targets = zero_targets

            weights = track["weights"]
            queries = track["queries"]
            vis_label = track["vis_label"]
            B, S, S1, M, C = targets.shape 
            # S = frames in sliding window
            # S1 = subtracks?
            # M = number of tracked points
            # C = dimension of tracks, e.g. (x, y)

            # print(f'- targets ({targets.shape})')
            # print(f'- S {S}, S1 {S1} and M {M}')

            # initializing track colors
            vector_colors = np.zeros((S, S1, M, 3)) # array to store RGB colors for each track across frames
            for s1 in range(S1):
                kf_stride = self.cfg_full.slam.kf_stride # controlling how frequently new keyframes are added
                fid_norm = ((fid // kf_stride + s1) % T) / T # normalizing fid between 0 and 1 to determine  color
                color = np.array(self.color_map(fid_norm)[:3]) * 255 # maps normalized fid to a color
                vector_colors[:, s1] = repeat(color, "c -> s m c", s=S, m=M)
                # vector_colors[t] = repeat()

            if "coords_vars" in track:
                variances = track["coords_vars"]

            # handle variances/uncertainty
            variances = track["coords_vars"] # contains the uncertainty values for each track point 
            var_mean = variances.mean(dim=1) # mean variance for each point across frames
            high_var_th = torch.quantile(var_mean, 0.9) # 90th percentile of the mean variance, used to filter points with high uncertainty
            high_mask = var_mean[0] > high_var_th # boolean mask identifying points with variance above the threshold
            # high_mask is used to handle dissapearing tracks, i.e., filter out tracks with too high uncertainty.
            variances = variances / variances.mean()  # normalized

            # extrating dynamic tracks
            # dyn_rgbs = res_video_dyn[fid - S : fid][None] #  frames corresponding to the dynamic track (fid - S to fid), with an extra batch dimension
            dyn_rgbs = combined_video[fid - S : fid][None] # customization
            dyn_tracks = targets.reshape(B, S, -1, C)[:, :, high_mask] # extracting points with high uncertainty (dynamic points) using high_mask
            dyn_vis_label = vis_label[:, :, high_mask] # visibility labels for those points
            dyn_colors = vector_colors.reshape(S, -1, 3)[ # colors for the dynamic points, reshaped to match the point indices
                :, high_mask.detach().cpu().numpy()
            ]

            # assigning yellow color to extracted tracks with high uncertainty
            dyn_color = mpl.colors.to_rgba("yellow")
            dyn_colors[..., 0] = dyn_color[0] * 255
            dyn_colors[..., 1] = dyn_color[1] * 255
            dyn_colors[..., 2] = dyn_color[2] * 255

            # extracting variances for dynamic tracks, but it is not used when drawing uncertain dynamic tracks
            dyn_var = (
                variances[:, :, high_mask].detach().cpu().numpy()
                if variances is not None
                else None
            )

            # print(f'- yellow dyn_tracks ({dyn_tracks.shape})')

            # print('- Call draw_tracks_on_video() with dynamic yellow parameters (no variance)')
            # drawing dynamic yellow tracks on video
            res_video = self.draw_tracks_on_video(
                video=dyn_rgbs, # visualizes the dynamic tracks on the selected frames (dyn_rgbs)
                tracks=dyn_tracks,
                visibility=dyn_vis_label,
                vector_colors=dyn_colors,
                variances=None,  # dyn_var
            )
            # res_video_dyn[fid - S : fid] = res_video # updates the corresponding frames in res_video_dyn with the drawn tracks
            combined_video[fid - S : fid] = res_video # customization

            variances = None # resetting variances

            # handling static tracks
            if "static_label" in track: 
                static_label = track["static_label"] # static_label is a tensor indicating which track points are considered static vs. dynamic
                # likely contains a per-point classification (1 = static, 0 = dynamic) ???

                # dyn_rgbs = res_video_dyn[fid - S : fid][None]
                dyn_rgbs = combined_video[fid - S : fid][None] # customization

                # check dynamic mask of the full track
                static_mask = static_label[0].float().mean(dim=0) < 0.5 # checks if the average confidence score for static points is low (< 0.5)
                # boolean mask (static_mask), where True represents dynamic points and False represents static points

                # extracting dynamic tracks
                dyn_tracks = targets.reshape(B, S, -1, C)[:, :, static_mask] # points classified as dynamic (static_mask is True).
                dyn_vis_label = vis_label[:, :, static_mask] # visibility information for dynamic points
                dyn_colors = vector_colors.reshape(S, -1, 3)[ # extracting corresponding colors
                    :, static_mask.detach().cpu().numpy()
                ]

                # print(f'- red dyn_tracks ({dyn_tracks.shape})')

                # assigning red color to dynamic tracks
                dyn_color = mpl.colors.to_rgba("red")
                dyn_colors[..., 0] = dyn_color[0] * 255
                dyn_colors[..., 1] = dyn_color[1] * 255
                dyn_colors[..., 2] = dyn_color[2] * 255

                # extracting variance for dynamic tracks
                dyn_var = (
                    variances[:, :, static_mask].detach().cpu().numpy()
                    if variances is not None
                    else None
                )

                # print('- Call draw_tracks_on_video() with dynamic red parameters (with variance)')
                # drawing dynamic tracks 
                # TODO: Are unertain yellow tracks drawn again in red?
                res_video = self.draw_tracks_on_video(
                    video=dyn_rgbs, # overlay dynamic tracks on the dyn_rgbs frames
                    tracks=dyn_tracks,
                    visibility=dyn_vis_label,
                    vector_colors=dyn_colors,
                    variances=dyn_var,
                )
                # res_video_dyn[fid - S : fid] = res_video # stores the updated frames back into res_video_dyn
                combined_video[fid - S : fid] = res_video # customization

                # extracting static tracks
                # rgbs = res_video_sta[fid - S : fid][None]
                rgbs = combined_video[fid - S : fid][None] # customization
                sta_tracks = targets.reshape(B, S, -1, C)[:, :, ~static_mask] # ~static_mask negates the mask
                sta_vis_label = vis_label[:, :, ~static_mask] # assign visibility labels
                sta_colors = vector_colors.reshape(S, -1, 3)[ # extract colors for static tracks
                    :, ~static_mask.detach().cpu().numpy()
                ]

                # use one color
                # color static tracks green
                sta_color = mpl.colors.to_rgba("lawngreen")
                sta_colors[..., 0] = sta_color[0] * 255
                sta_colors[..., 1] = sta_color[1] * 255
                sta_colors[..., 2] = sta_color[2] * 255

                # extract variance for static tracks
                sta_var = (
                    variances[:, :, ~static_mask].detach().cpu().numpy()
                    if variances is not None
                    else None
                )
                # print(f'- green sta_tracks ({sta_tracks.shape})')

                # print('- Call draw_tracks_on_video() with static green parameters (with variance)')
                # draw green static tracks 
                res_video = self.draw_tracks_on_video(
                    video=rgbs,
                    tracks=sta_tracks,
                    visibility=sta_vis_label,
                    vector_colors=sta_colors,
                    variances=sta_var,
                )
                # res_video_sta[fid - S : fid] = res_video
                combined_video[fid - S : fid] = res_video # customization

            else:
                # if static labels are not available it assumes all tracks are "static"
                # rgbs = res_video_sta[fid - S : fid][None]
                rgbs = combined_video[fid - S : fid][None] # customization
                # print('- Call draw_tracks_on_video() with general parameters (no variance)')
                # draw all tracks on res_video_sta without variance
                res_video = self.draw_tracks_on_video(
                    video=rgbs,
                    tracks=targets.reshape(B, S, -1, C),
                    visibility=vis_label,
                    vector_colors=vector_colors.reshape(S, -1, 3),
                )

                # res_video_sta[fid - S : fid] = res_video
                combined_video[fid - S : fid] = res_video # customization

        # res_video = torch.cat([res_video_sta, res_video_dyn], dim=-2)
        res_video = combined_video # customization

        # concatenates the static (res_video_sta) and dynamic (res_video_dyn) video representations along the width (dim=-2)
        # results in a single output video with both static and dynamic tracks visualized side by side

        return res_video[None].byte() # returns the final video with an added empty batch dimension and in byte format

    def save_video(self, filename, writer=None, step=0):
        video = self.draw_tracks_on_frames()

        # export video
        if writer is not None:
            writer.add_video(
                f"{filename}_pred_track",
                video.to(torch.uint8),
                global_step=step,
                fps=self.fps,
            )
        else:
            os.makedirs(self.save_dir, exist_ok=True)
            wide_list = list(video.unbind(1))
            wide_list = [wide[0].permute(1, 2, 0).cpu().numpy() for wide in wide_list]
            clip = ImageSequenceClip(wide_list[2:-1], fps=self.fps)

            # Write the video file
            save_path = os.path.join(self.save_dir, f"{filename}_pred_track.mp4")
            clip.write_videofile(save_path, codec="libx264", fps=self.fps, logger=None)

            print(f"Video saved to {save_path}")
