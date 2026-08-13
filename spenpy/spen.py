import torch
import torch.nn.functional as F
import math


def apply_fftshift_to_batch(batch, dim):
    # Apply fftshift to each element in the batch
    return torch.stack([torch.fft.fftshift(item, dim=dim) for item in batch])

def apply_fft_to_batch(batch, dim):
    # Apply ifft to each element in the batch
    return torch.stack([torch.fft.fft(item, dim=dim) for item in batch])

def fft_kspace_to_xspace_3d_batch(PreFFT, dim):
    return apply_fftshift_to_batch(apply_fft_to_batch(apply_ifftshift_to_batch(PreFFT, dim=dim), dim=dim), dim=dim)

def apply_ifftshift_to_batch(batch, dim):
    # Apply ifftshift to each element in the batch
    return torch.stack([torch.fft.ifftshift(item, dim=dim) for item in batch])

def apply_ifft_to_batch(batch, dim):
    # Apply ifft to each element in the batch
    return torch.stack([torch.fft.ifft(item, dim=dim) for item in batch])

def fft_xspace_to_kspace_3d_batch(PreFFT, dim):
    return apply_fftshift_to_batch(apply_ifft_to_batch(apply_ifftshift_to_batch(PreFFT, dim=dim), dim=dim), dim=dim)

def polyval2(p, x, y):
    device = p.device
    # Ensure x and y are tensors
    x = x.view(-1, 1)  # Reshape x to be a column vector
    y = y.view(1, -1)  # Reshape y to be a row vector
    
    lx = x.size(0)
    ly = y.size(1)
    lp = p.size(0)
    pts = lx * ly
    
    n = round((math.sqrt(1 + 8 * lp) - 3) / 2)
    
    # Create a grid of x and y coordinates
    x_grid = x.repeat(1, ly).view(-1)
    y_grid = y.repeat(lx, 1).view(-1)
    
    # Create Vandermonde matrix
    V = torch.ones((pts, lp))
    V = V.to(device)
    ordercolumn = 1
    for order in range(1, int(n) + 1):
        for ordercolumn in range(ordercolumn, ordercolumn + order):
            V[:, ordercolumn] = x_grid * V[:, ordercolumn - order]
        ordercolumn += 1
        V[:, ordercolumn] = y_grid * V[:, ordercolumn - order - 1]

    # Multiply Vandermonde matrix with polynomial coefficients
    z = torch.matmul(V, p.view(-1, 1))  # V * p
    z = z.view(ly, lx)  # Reshape to match the grid shape

    return z

def MathSinc(x):
    # MathSinc implementation (sinc function is sin(pi*x)/(pi*x))
    return torch.sinc(x / math.pi)

@torch.no_grad()
def calcSRMatrixApprox(MaxPhase, NumPixels, k, Partitions, b=None, ZeroThreshold=None):
    DefaultZeroThreshold = 10 * torch.finfo(torch.float32).eps

    # Handle the optional parameters
    if ZeroThreshold is None:
        ZeroThreshold = DefaultZeroThreshold

    if b is None:
        b = -(2 * MaxPhase / (Partitions[-1] - Partitions[0])**2) * Partitions[0]  # if b is not provided, calculate it

    aEffective = MaxPhase / (Partitions[-1] - Partitions[0])**2

    # If k is not defined, calculate it
    if k is None:
        k = -2 * aEffective * torch.arange(NumPixels).float()

    # Reshaping Partitions and k
    Partitions = Partitions.reshape(-1, 1)
    IdxPositions = (Partitions[:-1] + Partitions[1:]) / 2
    delta = Partitions[1:] - IdxPositions

    # Ensuring k has the correct shape
    k = k.reshape(-1, 1)
    NumKs = len(k)

    deltaMat = delta.view(1, -1).repeat(NumKs, 1)
    IdxPosMat = IdxPositions.view(1, -1).repeat(NumKs, 1)
    kMat = k.repeat(1, NumPixels)

    LinCoeffMat = (2 * aEffective * IdxPosMat + b + kMat)
    LinCoeff_x_delta_Mat = LinCoeffMat * deltaMat

    SincInput = LinCoeff_x_delta_Mat
    ExpInput = aEffective * IdxPosMat**2 + b * IdxPosMat + kMat * IdxPosMat
    HighOrder2 = 2 * ((LinCoeff_x_delta_Mat**2 - 2) * torch.sin(LinCoeff_x_delta_Mat) + 
                     2 * LinCoeff_x_delta_Mat * torch.cos(LinCoeff_x_delta_Mat)) / (LinCoeffMat**3)

    # Set small values of LinCoeffMat to ZeroThreshold
    ZeroLinCoeffMatIdxs = torch.abs(LinCoeffMat) < ZeroThreshold
    HighOrder2[ZeroLinCoeffMatIdxs] = (2 / 3) * deltaMat[ZeroLinCoeffMatIdxs]**3

    # Derivative calculation
    DerivativeOrder1 = 2 * 1j / LinCoeffMat**2 * (torch.sin(LinCoeff_x_delta_Mat) - LinCoeff_x_delta_Mat * torch.cos(LinCoeff_x_delta_Mat))

    # 2nd order calculation for A
    A = torch.exp(1j * ExpInput) * ((2 * deltaMat) * MathSinc(SincInput) + 1j * aEffective * HighOrder2)

    ADerivative = torch.exp(1j * ExpInput) * DerivativeOrder1

    PartitionsUsed = Partitions

    return A, ADerivative, IdxPositions, PartitionsUsed

@torch.no_grad()
def calcInvA(a_rad2cmsqr, LPE, NumPE, ShiftPE, SPENAcquireSign, ky1RelativePos, GaussRelativeWidth):
    MaxPhase = a_rad2cmsqr * LPE**2  # [rad]

    NumPixels = NumPE
    NumPixelsFinal = NumPE

    # Define positions of pixel borders.
    Partitions = SPENAcquireSign * torch.linspace(-LPE/2, LPE/2, NumPixels + 1) + ShiftPE / 10
    PartitionsFinal = SPENAcquireSign * torch.linspace(-LPE/2, LPE/2, NumPixelsFinal + 1) + ShiftPE / 10

    # Define ky sample positions.
    ky = -2 * SPENAcquireSign * a_rad2cmsqr * (torch.arange(NumPE).float()) * LPE / NumPE

    b = - ky[0] + -2 * a_rad2cmsqr * (Partitions[0] + (Partitions[1] - Partitions[0]) * ky1RelativePos)

    # Final value of b and ky.
    b = b.item()  # Convert b to a scalar (no problem here, as it's used in scalar form)
    
    # Ensure ky is kept as a tensor with NumPE elements
    ky = ky

    AFinal = calcSRMatrixApprox(MaxPhase, NumPixelsFinal, ky, PartitionsFinal, b)[0]

    # Generate Gaussian weighted Super-Resolution matrix.
    GaussWeightVar = (GaussRelativeWidth * math.pi * NumPixelsFinal**2 / MaxPhase)**2

    # Define y from given ky.
    yk = -(b + ky) / (2 * a_rad2cmsqr)

    # Define pixel centers
    yPixels = (PartitionsFinal[:-1] + PartitionsFinal[1:]) / 2

    # Calculate distances and translate to final pixel distances.
    DistMat = NumPixelsFinal / LPE * (yk.unsqueeze(1) - yPixels.unsqueeze(0))

    GaussWeight = torch.exp(-DistMat**2 / (2 * GaussWeightVar))

    AGaussWeighted = AFinal * GaussWeight

    # Define inverse of super-resolution matrix A to use.
    InvA = AGaussWeighted.conj().t()

    return InvA, AFinal


def check_data(data: torch.Tensor):
    if torch.is_complex(data):
        max_val = torch.abs(data).max()
        min_val = torch.abs(data).min()
    else:
        max_val = data.max()
        min_val = data.min()

    mean_val = data.mean()
    std_val = data.std()

    try:
        element = data[10, 11]
    except Exception as e:
        element = None

    return [mean_val.item(), max_val.item(), min_val.item(), std_val.item(), element.item()]


class spen:
    def __init__(self, L=[4, 4], acq_point=[256, 256], nseg=1, chirp_rvalue=120, tblip=128e-6, gamma_hz=4.2574e+3, device='cpu', noise_level=0.0):
        self.L = L
        self.acq_point = acq_point
        self.N = [acq_point[0], acq_point[1] * 16]
        self.x = torch.linspace(-L[0] / 2, L[0] / 2, self.N[0], device=device)
        self.y = torch.linspace(-L[1] / 2, L[1] / 2, self.N[1], device=device)
        self.Ydire_inhomo_coef = 0 * torch.tensor([0.001, 0.001, 0.00, 0.00], dtype=torch.float32, device=device) * L[1]
        self.nseg = nseg
        self.chirp_rvalue = chirp_rvalue
        self.tblip = tblip
        self.gamma_hz = gamma_hz
        self.device = device
        self.noise_level = noise_level
        self.spectrum = 0
        self.sw = 250000
        self.one_shot_num = acq_point[1] / nseg
        self.Ta = (acq_point[0] / self.sw + tblip) * self.one_shot_num
        self.chirp_tp = self.Ta / 2
        self.a_sign = -1

        self.procpar_struct = {
            'np': acq_point[0],
            'ne': 1,
            'nv': acq_point[1],
            'nseg': nseg,
            'nf': acq_point[1],
            'arraydim': 1,
            'Rvol': chirp_rvalue,
            'lpe': L[0],
            'lro': L[1],
            'ppe': 0,
            'Tp': self.chirp_tp,
            'Gchip': chirp_rvalue / self.chirp_tp / L[1] / gamma_hz,
        }

        self.rfwdth = self.procpar_struct['Tp']
        self.GPEe = self.procpar_struct['Gchip']
        self.alfa = self.a_sign * 2 * math.pi * gamma_hz * self.GPEe * self.rfwdth / (self.procpar_struct['lpe'])
        self.x_scale = torch.linspace(-L[0] / 2, L[0] / 2, self.N[0], device=device)
        self.y_scale = torch.linspace(-L[1] / 2, L[1] / 2, self.N[1], device=device)
    
    def get_InvA(self):
        return calcInvA(self.alfa, self.L[0], self.N[0], 0, self.a_sign, 0, 0.9)
    
    @torch.no_grad()
    def get_phase_map(self, H, noise_level=0.0):
        Hb = F.interpolate(H.unsqueeze(1), size=(self.acq_point[0], self.acq_point[1] * 16), mode='bilinear')
        Hb = Hb.squeeze(1)

        images = Hb

        KspaceTest = fft_xspace_to_kspace_3d_batch(fft_xspace_to_kspace_3d_batch(images, 0), 1)
        KspaceTestZero = torch.zeros([Hb.shape[0], int(self.acq_point[0] / 2), self.acq_point[1]], dtype=torch.complex64)
        lines = 10
        stard_idx_zero_x =  int(KspaceTestZero.shape[1] / 2) - lines
        end_idx_zero_x =  int(KspaceTestZero.shape[1] / 2) + lines
        stard_idx_zero_y =  int(KspaceTestZero.shape[2] / 2) - lines
        end_idx_zero_y =  int(KspaceTestZero.shape[2] / 2) + lines
        
        stard_idx_x =  int(KspaceTest.shape[1] / 2) - lines
        end_idx_x =  int(KspaceTest.shape[1] / 2) + lines
        stard_idx_y =  int(KspaceTest.shape[2] / 2) - lines
        end_idx_y =  int(KspaceTest.shape[2] / 2) + lines
        
        
        KspaceTestZero[:, stard_idx_zero_x : end_idx_zero_x, stard_idx_zero_y : end_idx_zero_y] = KspaceTest[:, stard_idx_x : end_idx_x, stard_idx_y : end_idx_y]

        map_img = fft_kspace_to_xspace_3d_batch(fft_kspace_to_xspace_3d_batch(KspaceTestZero, 0), 1)
        map_img = torch.abs(map_img) / torch.max(torch.abs(map_img))
        EvenOddLinear = noise_level * 2 * (torch.rand(1, 1, device=self.device) - 0.5)
        EvenOddConstant = noise_level * 2 * (torch.rand(1, 1, device=self.device) - 0.5)

        linspace_vals = torch.linspace(-self.L[0] / 2, self.L[0] / 2, self.acq_point[0], device=self.device)

        phase_map = (torch.ones((self.acq_point[1] // 2 // self.nseg, 1), device=self.device) * (2 * torch.pi * (EvenOddLinear[0] * linspace_vals + EvenOddConstant[0]))) + 2 * torch.pi * map_img
        return phase_map
    
    @torch.no_grad()
    def sim(self, H, return_phase_map=False, return_good_lr_image=False): # <-- Added return_good_lr_image flag
        Hb = F.interpolate(H.permute(0, 2, 1).unsqueeze(1), size=(self.acq_point[0], self.acq_point[1] * 16), mode='bilinear')
        Hb = Hb.squeeze(1)
        seg_random = 0.00 * self.L[1] / 100 * torch.rand(1).item()
        phase_map_ideal = torch.zeros(Hb.shape[0], self.nseg, self.acq_point[0], int(self.acq_point[1] / self.nseg))
    
        good_lr_image = torch.zeros((Hb.shape[0], self.acq_point[1], self.acq_point[0]), dtype=torch.complex64, device=self.device)
        
        for k in range(self.nseg):
            p = 0 * torch.rand(7, device=self.device)

            start = -self.L[1] / 2 + k * self.L[1] / self.acq_point[1] + seg_random
            step = self.L[1] / (self.acq_point[1] / self.nseg)
            end = start + (self.acq_point[1] / self.nseg - 1) * step
            temp_yacq = torch.arange(start, end + step, step, device=self.device)
            
            b0y = (self.Ydire_inhomo_coef[0] * self.y**0 + 
                   self.Ydire_inhomo_coef[1] * self.y**1 + 
                   self.Ydire_inhomo_coef[2] * self.y**2 + 
                   self.Ydire_inhomo_coef[3] * self.y**3)
            
            b0acq = (self.Ydire_inhomo_coef[0] * temp_yacq**0 + 
                     self.Ydire_inhomo_coef[1] * temp_yacq**1 + 
                     self.Ydire_inhomo_coef[2] * temp_yacq**2 + 
                     self.Ydire_inhomo_coef[3] * temp_yacq**3)
            
            y_grid, temp_yacq_grid = torch.meshgrid(self.y, temp_yacq,  indexing="ij")
            b0y_grid, b0acq_grid = torch.meshgrid(b0y, b0acq,  indexing="ij")
            part1 = (y_grid + b0y_grid) - (temp_yacq_grid + b0acq_grid)
            part2 = temp_yacq_grid + b0acq_grid
            exp_term = torch.exp(1j * self.alfa * (part1**2 - part2**2))
            exp_term = exp_term.to(torch.complex64)
            spin_density = Hb
            motion_imag = spin_density
            motion_imag = motion_imag.to(torch.complex64)
            temprxyacq = torch.matmul(motion_imag.view(H.shape[0], self.acq_point[0], self.acq_point[1] * 16), exp_term)
            
            good_lr_image[:, k::self.nseg, :] = temprxyacq.permute(0, 2, 1)
            motion_phase_map = polyval2(p, self.y, self.x).unsqueeze(0)
            motion_phase_map = F.interpolate(motion_phase_map.unsqueeze(0), size=temprxyacq.shape[1:], mode='bilinear', align_corners=False).squeeze(1)
            motion_phase_map_complex = torch.exp(1j * motion_phase_map)
            phase_map_ideal[:, k, :, :] = motion_phase_map
            temprxyacq = temprxyacq * motion_phase_map_complex
            temprxyacq = temprxyacq.permute(0, 2, 1)

            temprxyacq = fft_xspace_to_kspace_3d_batch(temprxyacq, 1)      
            img = fft_kspace_to_xspace_3d_batch(temprxyacq, 1)
            im_odd = img[:, 0::2, :] 
            im_even = img[:, 1::2, :] 
            
            if self.nseg % 2 == 1:
                if k % 2 == 0:
                    images = H

                    KspaceTest = fft_xspace_to_kspace_3d_batch(fft_xspace_to_kspace_3d_batch(images, 0), 1)
                    KspaceTestZero = torch.zeros([Hb.shape[0], int(self.acq_point[0] / 2), self.acq_point[1]], dtype=torch.complex64, device=self.device)
                    lines = 10
                    stard_idx_zero_x =  int(KspaceTestZero.shape[1] / 2) - lines
                    end_idx_zero_x =  int(KspaceTestZero.shape[1] / 2) + lines
                    stard_idx_zero_y =  int(KspaceTestZero.shape[2] / 2) - lines
                    end_idx_zero_y =  int(KspaceTestZero.shape[2] / 2) + lines
                    
                    stard_idx_x =  int(KspaceTest.shape[1] / 2) - lines
                    end_idx_x =  int(KspaceTest.shape[1] / 2) + lines
                    stard_idx_y =  int(KspaceTest.shape[2] / 2) - lines
                    end_idx_y =  int(KspaceTest.shape[2] / 2) + lines
                    
                    
                    KspaceTestZero[:, stard_idx_zero_x : end_idx_zero_x, stard_idx_zero_y : end_idx_zero_y] = KspaceTest[:, stard_idx_x : end_idx_x, stard_idx_y : end_idx_y]

                    map_img = fft_kspace_to_xspace_3d_batch(fft_kspace_to_xspace_3d_batch(KspaceTestZero, 0), 1)
                    map_img = torch.abs(map_img) / torch.max(torch.abs(map_img))
                    emap = torch.cos(2 * math.pi * map_img) + 1j * torch.sin(2 * math.pi * map_img)
                    EvenOddLinear = 1 * (torch.rand(1, 1, device=self.device) - 0.5)
                    EvenOddConstant = 1 * (torch.rand(1, 1, device=self.device) - 0.5)

                    linspace_vals = torch.linspace(-self.L[0] / 2, self.L[0] / 2, self.acq_point[0], device=self.device)

                    phase_map = (torch.ones((self.acq_point[1] // 2 // self.nseg, 1), device=self.device) * (2 * torch.pi * (EvenOddLinear[0] * linspace_vals + EvenOddConstant[0]))) + 2 * torch.pi * map_img
                    im_even = im_even * emap
                    img[:, 1::2, :] = im_even
    
            temprxyacq = fft_xspace_to_kspace_3d_batch(img, 1)
            temprxyacq = fft_kspace_to_xspace_3d_batch(temprxyacq, 1)
            motion_phase_map1 = polyval2(p, self.y, self.x).unsqueeze(0)
            motion_phase_map1 = F.interpolate(motion_phase_map1.unsqueeze(0), size=temprxyacq.shape[1:], mode='bilinear').squeeze(1)
            motion_phase_map_complex1 = torch.exp(-0j * motion_phase_map1)
            temprxyacq = temprxyacq * motion_phase_map_complex1
            temprxyacq = fft_xspace_to_kspace_3d_batch(temprxyacq, 1)
            temprxyacq = temprxyacq / torch.max(torch.abs(temprxyacq))
            
            final_rxyacq = temprxyacq + self.noise_level * torch.randn_like(temprxyacq)
            final_rxyacq_ROFFT = fft_kspace_to_xspace_3d_batch(final_rxyacq, 1)

        # Calculate max absolute value per slice in the batch for scaling
        max_vals = torch.max(torch.abs(good_lr_image).view(good_lr_image.shape[0], -1), dim=1)[0]
        good_lr_image = good_lr_image / max_vals.view(-1, 1, 1)

        # --- NEW: 4. Adjusted Return statements ---
        if return_phase_map and return_good_lr_image:
            return final_rxyacq_ROFFT, phase_map, good_lr_image
        elif return_good_lr_image:
            return final_rxyacq_ROFFT, good_lr_image
        elif return_phase_map:
            return final_rxyacq_ROFFT, phase_map
        else:
            return final_rxyacq_ROFFT


from spenpy.sim.spen_sim import SpenSim as _ConfigurableSpenSim


class spen(_ConfigurableSpenSim):
    """Backward-compatible name for the YAML-configurable simulator.

    Existing code can keep using ``from spenpy.spen import spen`` and the
    historical constructor arguments.  New code may pass ``config=...`` or use
    ``spen.from_yaml(...)`` through the inherited :class:`SpenSim` API.
    """

    pass
