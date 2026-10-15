"""Independent exact audit of the general construction and complex rotations.

Run: python3 -m checks.verify_full_construction
This script does not import or call the numerical HSVD implementation.
"""
import json
from pathlib import Path
import sympy as s


def verify():

    # All arithmetic here is exact, independently of the implementation.
    J=s.diag(1,1,1,1,-1,-1,-1,-1)
    base=s.zeros(8,4)
    base[0,0]=base[4,1]=1
    base[1,2]=base[5,2]=1/s.sqrt(2)
    base[2,3]=base[6,3]=1/s.sqrt(2)
    R=s.eye(4)
    for a,b,c,t in [(0,2,s.Rational(3,5),s.Rational(4,5)),(1,3,s.Rational(5,13),s.Rational(12,13))]:
        G=s.eye(4);G[a,a]=G[b,b]=c;G[a,b]=G[b,a]=s.I*t
        R=R*G
    Vr=base*R
    S=s.diag(7,5,3,2)
    H=s.simplify(Vr.H*J*Vr)
    C=s.simplify(S*H*S)
    Ur=s.eye(5)[:,:4]
    A=Ur*S*Vr.H
    B=A.H*A*J

    def zero(M):
        return all(s.simplify(x)==0 for x in M)

    def normalize(v):
        return v/s.sqrt(s.simplify((v.H*v)[0]))

    blocks={1:[], -1:[],0:[]};lam={1:[],-1:[]}
    for val,mult,vecs in C.eigenvects():
        sign=int(s.sign(val))
        blocks[sign]+=[normalize(v) for v in vecs]
        if sign:lam[sign]+=[val]*len(vecs)
    Zp,Zm,Z0=[s.Matrix.hstack(*blocks[k]) for k in [1,-1,0]]
    # Mix the kernel basis to exercise joint, off-diagonal Gram normalization.
    Z0=Z0*s.Matrix([[s.Rational(3,5),s.I*s.Rational(4,5)],[s.I*s.Rational(4,5),s.Rational(3,5)]])
    Z=Zp.row_join(Zm).row_join(Z0)
    assert zero(Z.H*Z-s.eye(4))
    N=S*Z0
    assert zero(H*N)
    assert not zero(H*Z0)  # ker H and ker C are genuinely different here.
    E=Vr*N;F=J*Vr*S.inv()*Z0
    assert zero(B*E) and zero(B*F-E)
    assert zero(E.H*J*E) and zero(E.H*J*F-s.eye(2))
    D=s.simplify(F.H*J*F)
    assert D[0,1]!=0
    G=F-E*D/2
    Vv=E/2+G;Vw=E/2-G
    assert zero(G.H*J*G)
    assert zero(Vv.H*J*Vv-s.eye(2))
    assert zero(Vw.H*J*Vw+s.eye(2))
    assert zero(Vv.H*J*Vw)
    P=s.diag(*[s.sqrt(x) for x in lam[1]])
    Q=s.diag(*[s.sqrt(-x) for x in lam[-1]])
    Xp=Vr*S*Zp*P.inv();Xm=Vr*S*Zm*Q.inv()
    T=Xp.row_join(Xm).row_join(Vv).row_join(Vw)
    W=s.eye(8)[:,[3,7]]
    assert zero(T.H*J*W)
    assert zero(W.H*J*W-s.diag(1,-1))
    assert zero(A*J*W)
    V=Xp.row_join(Vv).row_join(W[:,0]).row_join(Xm).row_join(Vw).row_join(W[:,1])
    U=(Ur*Z).row_join(s.eye(5)[:,4])
    Sigma=s.zeros(5,8)
    Sigma[0,0]=P[0,0];Sigma[1,4]=Q[0,0]
    Sigma[2,1]=Sigma[2,5]=Sigma[3,2]=Sigma[3,6]=1
    assert zero(U.H*U-s.eye(5))
    assert zero(V.H*J*V-J)
    assert zero(U*Sigma*V.H-A)
    # Chain theorem also works for an arbitrary nonorthonormal kernel basis.
    Nb=N*s.Matrix([[2,1+s.I],[0,3]])
    Eb=Vr*Nb;Fb=J*Vr*S.inv()**2*Nb
    assert zero(B*Eb) and zero(B*Fb-Eb)
    assert s.Matrix.hstack(Eb,Fb).rank()==4

    # Formal complex phase: conjugate(phase)=1/phase on the unit circle.
    a,d,b,c,t=s.symbols('a d b c t', real=True)
    phase=s.symbols('phase', nonzero=True)
    for h in (1,-1):
        rotation=s.Matrix([[c,phase*t],[-h*t/phase,c]])
        adjoint=s.Matrix([[c,-h*phase*t],[t/phase,c]])
        metric=s.diag(1,h)
        gram=s.Matrix([[a,b*phase],[b/phase,d]])
        assert zero(adjoint*metric*rotation-(c*c+h*t*t)*metric)
        cross=(adjoint*gram*rotation)[0,1]
        assert s.simplify(cross-phase*(b*(c*c-h*t*t)+(a-h*d)*c*t))==0
    return dict(status='passed', arithmetic='exact SymPy',
                complex_constructions=1, isotropic_chains=2,
                nonorthonormal_kernel_basis=True, off_diagonal_gram=True,
                distinct_H_C_kernels=True, completion_signature=[1,1],
                complex_rotation_identities=2)


if __name__ == '__main__':
    result=verify()
    path=Path(__file__).resolve().parents[1]/'results/audit/exact_construction.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
