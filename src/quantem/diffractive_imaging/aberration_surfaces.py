import math

import torch

aberration_terms = [
    (1, 0, "C10", None),
    (1, 2, "C12", "phi12"),
    (2, 1, "C21", "phi21"),
    (2, 3, "C23", "phi23"),
    (3, 0, "C30", None),
    (3, 2, "C32", "phi32"),
    (3, 4, "C34", "phi34"),
    (4, 1, "C41", "phi41"),
    (4, 3, "C43", "phi43"),
    (4, 5, "C45", "phi45"),
    (5, 0, "C50", None),
    (5, 2, "C52", "phi52"),
    (5, 4, "C54", "phi54"),
    (5, 6, "C56", "phi56"),
]


def chi_grad_hessian_nm(
    qx, qy, lam, n, m, C_a, C_b, include_gradient=True, include_hessian=True
):
    """ """
    q2 = qx**2 + qy**2
    q = torch.sqrt(q2)

    phi = torch.atan2(qy, qx)
    pi = math.pi

    match (n, m):
        case (1, 0):
            chi = pi * lam * q2 * C_a

            if include_gradient:
                dx = 2 * pi * lam * C_a * qx
                dy = 2 * pi * lam * C_a * qy
                if include_hessian:
                    hxx = 2 * pi * lam * C_a
                    hxy = 0.0
                    hyy = hxx

        case (1, 2):
            chi = pi * lam * (C_a * qx**2 - C_a * qy**2 + 2 * C_b * qx * qy)

            if include_gradient:
                dx = 2 * pi * lam * (C_a * qx + C_b * qy)
                dy = 2 * pi * lam * (C_b * qx - C_a * qy)
                if include_hessian:
                    hxx = 2 * pi * lam * C_a
                    hxy = 2 * pi * lam * C_b
                    hyy = -hxx

        case (2, 1):
            chi = 2 * pi / 3 * lam**2 * q2 * (C_a * qx + C_b * qy)

            if include_gradient:
                dx = (
                    2
                    * pi
                    / 3
                    * lam**2
                    * (3 * qx * (C_a * qx + C_b * qy) + qy * (C_a * qy - C_b * qx))
                )
                dy = (
                    2
                    * pi
                    / 3
                    * lam**2
                    * (3 * qy * (C_a * qx + C_b * qy) - qx * (C_a * qy - C_b * qx))
                )
                if include_hessian:
                    hxx = 4 * pi / 3 * lam**2 * (3 * C_a * qx + C_b * qy)
                    hxy = 4 * pi / 3 * lam**2 * (C_a * qy + C_b * qx)
                    hyy = 4 * pi / 3 * lam**2 * (C_a * qx + 3 * C_b * qy)

        case (2, 3):
            cos_phi = torch.cos(3 * phi)
            sin_phi = torch.sin(3 * phi)
            chi = 2 * pi / 3 * lam**2 * q**3.0 * (C_a * cos_phi + C_b * sin_phi)

            if include_gradient:
                dx = (
                    2
                    * pi
                    * lam**2
                    * q
                    * (
                        qx * (C_a * cos_phi + C_b * sin_phi)
                        + qy * (C_a * sin_phi - C_b * cos_phi)
                    )
                )
                dy = (
                    2
                    * pi
                    * lam**2
                    * q
                    * (
                        qy * (C_a * cos_phi + C_b * sin_phi)
                        - qx * (C_a * sin_phi - C_b * cos_phi)
                    )
                )
                if include_hessian:
                    hxx = (
                        4
                        * pi
                        * lam**2
                        / q
                        * (
                            C_a * qx**2 * cos_phi
                            + 2 * C_a * qx * qy * sin_phi
                            - C_a * qy**2 * cos_phi
                            + C_b * qx**2 * sin_phi
                            - 2 * C_b * qx * qy * cos_phi
                            - C_b * qy**2 * sin_phi
                        )
                    )
                    hxx[0, 0] = 0.0
                    hxy = (
                        4
                        * pi
                        * lam**2
                        / q
                        * (
                            -C_a * qx**2 * sin_phi
                            + 2 * C_a * qx * qy * cos_phi
                            + C_a * qy**2 * sin_phi
                            + C_b * qx**2 * cos_phi
                            + 2 * C_b * qx * qy * sin_phi
                            - C_b * qy**2 * cos_phi
                        )
                    )
                    hxy[0, 0] = 0.0
                    hyy = -hxx.clone()

        case (3, 0):
            chi = pi / 2 * lam**3 * C_a * q2**2

            if include_gradient:
                dx = 2 * pi * lam**3 * C_a * qx * q2
                dy = 2 * pi * lam**3 * C_a * qy * q2
                if include_hessian:
                    hxx = 2 * pi * lam**3 * C_a * (3 * qx**2 + qy**2)
                    hxy = 4 * pi * lam**3 * C_a * qx * qy
                    hyy = 2 * pi * lam**3 * C_a * (qx**2 + 3 * qy**2)

        case _:
            raise NotImplementedError(
                f"Chi gradient/hessian for (n, m) = ({n}, {m}) not implemented."
            )

    if include_gradient:
        if include_hessian:
            return chi, dx, dy, hxx, hxy, hyy
        else:
            return chi, dx, dy
    return (chi,)


def chi_taylor_expansion(
    qxa,
    qya,
    wavelength,
    rotation_angle,
    coefs,
    include_gradient=True,
    include_hessian=True,
):
    """ """

    # Passive rotation of the coordinate grid
    cos_a = math.cos(-rotation_angle)
    sin_a = math.sin(-rotation_angle)
    qx_rot = qxa * cos_a + qya * sin_a
    qy_rot = -qxa * sin_a + qya * cos_a

    chi = torch.zeros_like(qxa)

    if include_gradient:
        dx = torch.zeros_like(qxa)
        dy = torch.zeros_like(qxa)
        if include_hessian:
            hxx = torch.zeros_like(qxa)
            hxy = torch.zeros_like(qxa)
            hyy = torch.zeros_like(qxa)

            arrays = [chi, dx, dy, hxx, hxy, hyy]
        else:
            arrays = [chi, dx, dy]
    else:
        arrays = [chi]

    for n, m, Cname, phiname in aberration_terms:
        if Cname not in coefs:
            continue

        C_nm = coefs[Cname]
        phi_nm = coefs.get(phiname, 0.0)

        C_a = C_nm * math.cos(m * phi_nm)
        C_b = C_nm * math.sin(m * phi_nm)

        values = chi_grad_hessian_nm(
            qx_rot,
            qy_rot,
            wavelength,
            n,
            m,
            C_a,
            C_b,
            include_gradient=include_gradient,
            include_hessian=include_hessian,
        )

        for array, value in zip(arrays, values):
            array += value

    return arrays
