# 3a. Lie groups, Lie algebras and generators, from the ground up

*Chapter 3 states in one line that "the invertible matrices form a Lie group, its Lie algebra is the tangent space at the identity, and the exponential maps one to the other." Chapter 5 then builds the whole composition story on the bracket $[X,Y]$. This chapter slows that line down. It builds each object from a small matrix you can compute by hand, and only names the abstraction after you have already used it.*

> **Where this sits.** This is a companion to [Chapter 3](03-why-exp-of-m.md) (why $A = \exp(M)$) and a prerequisite for [Chapter 5](05-composition.md) (composition and BCH). Every numerical claim below is re-computed by [`checks/verify_lie_examples.py`](checks/verify_lie_examples.py), and section numbers in that script match the sections here.

---

## 0. Recap: the one line we are unpacking

Here is the action-operator pipeline as the earlier chapters left it:

$$c \enspace \xrightarrow{\text{policy}}\enspace \alpha \enspace \longrightarrow\enspace M = \sum_i \alpha_i B_i \enspace \longrightarrow\enspace A = \exp(M) \enspace \longrightarrow\enspace z' = A z .$$

Here is what each symbol means:

- $c$ is the intervention (a gene knockdown, say).
- $\alpha = (\alpha_1, \dots, \alpha_k)$ are the coefficients the policy network outputs.
- $B_i$ are fixed basis matrices, each of size $D \times D$, where $D$ is the latent dimension (256 in this project).
- $M$ is the **generator**: a $D \times D$ matrix built as a weighted sum of the basis matrices.
- $\exp(M)$ is the **matrix exponential** (defined in §3).
- $A$ is the **operator** the generator produces, also $D \times D$.
- $z$ is a cell's latent vector before the intervention, and $z'$ is the vector after it.

Chapter 3 described $M$ as living in a flat space, the "Lie algebra", and $A$ as living in a curved one, the "Lie group". This chapter shows what that means, using the smallest example where every step can be checked by hand: turning a clock hand.

---

## 1. The story starts with a clock hand

### 1.1 The worked example: rotating the plane

Take a point in the plane, written as a column vector $z = (z_1, z_2)^\top$. To rotate it counter-clockwise by an angle $\theta$ (in radians), multiply it by

$$R(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}.$$

To check this, rotate the point $(1, 0)$, which sits at 3 o'clock, by $\theta = \pi/2$ (a quarter turn). Since $\cos(\pi/2) = 0$ and $\sin(\pi/2) = 1$,

$$R(\pi/2)\begin{pmatrix}1\\0\end{pmatrix} = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}\begin{pmatrix}1\\0\end{pmatrix} = \begin{pmatrix}0\\1\end{pmatrix}.$$

The point moved to 12 o'clock, as a quarter turn should.

### 1.2 Four facts you already believe about rotations

Each fact below is a one-line calculation.

1. **Doing two rotations in a row is a rotation.** Rotating by $a$ and then by $b$ is the same as rotating by $a + b$:
   $$R(b)R(a) = R(a + b).$$
   (Multiply the matrices out and you get the angle-addition formulas for $\cos(a+b)$ and $\sin(a+b)$.)
2. **"Do nothing" is a rotation.** $R(0) = I$, the identity matrix.
3. **Every rotation can be undone.** $R(-\theta)$ undoes $R(\theta)$, because $R(-\theta)R(\theta) = R(0) = I$.
4. **Grouping doesn't matter.** $(R_3 R_2) R_1 = R_3 (R_2 R_1)$. This holds because matrix multiplication is associative.

### 1.3 The fact that makes rotations awkward for a network

Now **add** two rotations instead of composing them. Take $R(0) = I$ and $R(\pi)$, which is a half-turn equal to $-I$:

$$R(0) + R(\pi) = I + (-I) = \begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix}.$$

The sum is the zero matrix. That is not a rotation, and it is not even invertible. **Adding two rotations can take you out of the set of rotations.** Rotation matrices can be multiplied but not added. Their natural operation is composition.

This matters for us because a neural network's output layer produces an arbitrary vector of real numbers, and it moves those numbers by adding gradient steps to them. Both of those are operations of a flat space, where adding and scaling are always allowed. The set of rotations is not such a space.

### 1.4 What the set of rotations looks like

Every $2 \times 2$ rotation is determined by a single angle, and the angles $\theta$ and $\theta + 2\pi$ give the same matrix. So the set of rotations is a **circle**: a one-dimensional curved object. Each point on it is a rotation, and the identity $I$ is one particular point.

Mathematicians call this set $SO(2)$, the *special orthogonal group* in 2 dimensions:
- **orthogonal** means $R^\top R = I$, so lengths are preserved;
- **special** means $\det R = +1$, so there are no mirror flips.

---

## 2. Naming what we just saw: a Lie group

The clock hand had two kinds of structure at once.

**Algebraic structure (a *group*).** A **group** is a set $G$ with a way of combining two elements (here, matrix multiplication) such that:
- combining two elements always gives another element of $G$ (*closure*, fact 1);
- there is an identity element that changes nothing (fact 2);
- every element has an inverse (fact 3);
- combining is associative (fact 4).

**Geometric structure (a *smooth manifold*).** The set is also a smooth curved space. Near any point, it looks like ordinary flat $\mathbb{R}^n$ for some $n$, the way a small patch of the Earth's surface looks like a flat plane. The number $n$ is the **dimension** of the group. For $SO(2)$ it is $1$, because one angle is enough. The group operations (multiplying, inverting) are smooth functions on this space.

> **Definition.** A **Lie group** is a group that is also a smooth manifold, with smooth multiplication and inversion. (It is named after Sophus Lie, who studied continuous symmetries of differential equations in the 1870s.)

Here are the Lie groups this series touches. Each is a set of $D \times D$ matrices whose combining operation is matrix multiplication:

| group | members | dimension | in plain words |
|---|---|---|---|
| $SO(2)$ | $2\times 2$ rotations | 1 | turn the plane |
| $\mathbb{R}_{>0}$ under multiplication | positive numbers, which are $1 \times 1$ matrices | 1 | rescale a single number |
| $SO(3)$ | $3\times 3$ rotations | 3 | turn a rigid body in space |
| $SE(2)$ | rigid motions of the plane | 3 | move and turn a car on a parking lot |
| $SL(2)$ | $2\times 2$, determinant 1 | 3 | area-preserving linear maps |
| $GL(D)$ | all invertible $D\times D$ | $D^2$ | any reversible linear map; $D = 256$ is the project's operator space |
| $GL^+(D)$ | invertible with $\det > 0$ | $D^2$ | the part of $GL(D)$ reachable from $I$ without crossing a singular matrix |

$GL(D)$ is the *general linear group*. Its "hole", as Chapter 3 calls it, is the set of singular matrices (those with $\det = 0$). That set separates the matrices with positive determinant from those with negative determinant, so $GL(D)$ is in two disconnected pieces and $GL^+(D)$ is the piece containing $I$.

---

## 3. Zooming in at the identity: the generator

### 3.1 The worked example: the velocity of a turning clock hand

Go back to the circle of rotations and stand at the identity, where $\theta = 0$. Ask: **if I start turning, which direction am I moving in?** In calculus terms, this is the derivative of $R(\theta)$ with respect to $\theta$, evaluated at $\theta = 0$.

Differentiate each entry ($\tfrac{d}{d\theta}\cos\theta = -\sin\theta$ and $\tfrac{d}{d\theta}\sin\theta = \cos\theta$):

$$\frac{dR}{d\theta} = \begin{pmatrix} -\sin\theta & -\cos\theta \\ \cos\theta & -\sin\theta \end{pmatrix} \quad\Longrightarrow\quad \left.\frac{dR}{d\theta}\right|_{\theta=0} = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix} \enspace \equiv\enspace J .$$

This matrix $J$ is the **generator of rotation**. It is not a rotation itself: $J$ is the velocity at which a rotation starts.

To see that it really is a velocity, apply $J$ to a point $z = (z_1, z_2)^\top$:

$$J z = \begin{pmatrix} -z_2 \\ z_1 \end{pmatrix}.$$

For $z = (1, 0)$ (3 o'clock), $Jz = (0, 1)$, which points straight up. For $z = (0, 1)$ (12 o'clock), $Jz = (-1, 0)$, which points left. In each case $Jz$ is perpendicular to $z$ and points counter-clockwise. **$J$ assigns to every point the velocity of a counter-clockwise spin.** It is a *velocity field*, an arrow attached to every point of the plane.

### 3.2 The flow that the velocity field produces

A velocity field defines motion. Each point follows its arrows. Written as a differential equation:

$$\dot z(t) = J z(t).$$

Here $z(t)$ is the point's position at time $t$, and $\dot z(t) = dz/dt$ is its velocity. The equation says: **at every moment, move with the velocity the field gives at your current position.** The solution is that each point travels around a circle at one radian per unit time. After time $\theta$, the point has been rotated by $R(\theta)$.

So we have two descriptions of one motion:
- **the generator** $J$, which says how things move *right now*, everywhere;
- **the group element** $R(\theta)$, which says where everything *ends up* after time $\theta$.

### 3.3 The Lie algebra of $SO(2)$

Every possible starting velocity at the identity is a multiple of $J$: $\omega J$ for some real number $\omega$. You can spin slowly, quickly, or backwards ($\omega < 0$), but always about the same centre. The set $\{\omega J : \omega \in \mathbb{R}\}$ is written $\mathfrak{so}(2)$, the **Lie algebra** of $SO(2)$. (By convention, a lowercase Fraktur letter names the algebra of the group with the matching capital letter.)

$\mathfrak{so}(2)$ is a **line**: a flat, one-dimensional vector space. You can add and scale its elements freely: $2J + 3J = 5J$ is still a valid velocity. Geometrically, it is the line that touches the circle of rotations at the identity, the **tangent line** there.

> **Picture to hold onto.** The group is a curved surface. The algebra is the flat plane that touches it at the identity. Here the "surface" is a circle and the "plane" is a line, but the picture scales up.

---

## 4. Getting back onto the circle: the exponential

### 4.1 The question

We have a velocity $\theta J$ in the flat algebra. We want the group element it produces after one unit of time. Chapter 3 says the answer is $\exp(\theta J)$. Let us compute it and check that it gives $R(\theta)$.

### 4.2 Reminder: the scalar case

For an ordinary number $a$, the equation $\dot x = a x$ has the solution $x(t) = e^{at} x(0)$, and the exponential has the power series

$$e^{a} = 1 + a + \frac{a^2}{2!} + \frac{a^3}{3!} + \cdots = \sum_{n=0}^{\infty} \frac{a^n}{n!}.$$

The **matrix exponential** uses the same series, with the number $a$ replaced by a square matrix $M$ and $1$ replaced by the identity matrix $I$:

$$\exp(M) = I + M + \frac{M^2}{2!} + \frac{M^3}{3!} + \cdots = \sum_{n=0}^{\infty} \frac{M^n}{n!}.$$

Here $M^n$ means $M$ multiplied by itself $n$ times, and $M^0 = I$. The series converges for every square matrix. It solves $\dot z = M z$ for the same reason the scalar version solves $\dot x = ax$: differentiating $\exp(tM)$ term by term gives $M \exp(tM)$.

### 4.3 The worked computation: $\exp(\theta J)$

The key fact is what happens when $J$ is multiplied by itself:

$$J^2 = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}\begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix} = \begin{pmatrix} -1 & 0 \\ 0 & -1 \end{pmatrix} = -I.$$

Two quarter-turn velocities compound into a sign flip. ($J$ behaves like the imaginary unit $i$, with $i^2 = -1$.) The powers of $J$ then repeat with period four:

$$J^0 = I,\quad J^1 = J,\quad J^2 = -I,\quad J^3 = -J,\quad J^4 = I,\ \dots$$

Substitute these into the series, writing $(\theta J)^n = \theta^n J^n$:

$$\exp(\theta J) = I + \theta J - \frac{\theta^2}{2!} I - \frac{\theta^3}{3!} J + \frac{\theta^4}{4!} I + \frac{\theta^5}{5!}J - \cdots$$

Now collect the terms that multiply $I$ and the terms that multiply $J$:

$$\exp(\theta J) = \underbrace{\Big(1 - \frac{\theta^2}{2!} + \frac{\theta^4}{4!} - \cdots\Big)}_{\cos\theta} I \enspace +\enspace \underbrace{\Big(\theta - \frac{\theta^3}{3!} + \frac{\theta^5}{5!} - \cdots\Big)}_{\sin\theta} J.$$

The two brackets are the Taylor series of $\cos\theta$ and $\sin\theta$. So

$$\exp(\theta J) = \cos\theta I + \sin\theta J = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} = R(\theta). \enspace \checkmark$$

> **What just happened.** We started on a flat line ($\theta J$, which can be any real multiple of $J$) and the exponential wrapped it around the circle of rotations. Every output is a valid rotation, whatever $\theta$ is. That is the property the action operator relies on: **the network outputs a point in the flat algebra, and $\exp$ guarantees a valid group element.**

### 4.4 Composition becomes addition (for this group)

Fact 1 from §1.2, $R(b)R(a) = R(a+b)$, now reads

$$\exp(bJ)\exp(aJ) = \exp\big((a + b) J\big).$$

Multiplying in the group corresponds to adding in the algebra. This is the matrix version of $e^a e^b = e^{a+b}$. It holds here because every element of $\mathfrak{so}(2)$ is a multiple of the same $J$, and any two such multiples commute. The rest of this chapter is about what happens when generators *don't* commute.

### 4.5 A second worked example: rescaling a number

The simplest Lie group of all is the positive numbers under multiplication, $\mathbb{R}_{>0}$, thought of as $1 \times 1$ matrices $(g)$:
- the identity is $1$;
- the inverse of $g$ is $1/g$;
- the algebra is all of $\mathbb{R}$ (any growth rate $m$, positive or negative);
- the exponential is the ordinary $e^m$, which is always positive.

The algebra-to-group map $m \mapsto e^m$ turns addition into multiplication, and its inverse, $\log$, turns multiplication back into addition. The action operator applies the same idea to whole matrices. The network outputs a log-scale quantity freely, and exponentiating it guarantees the output is valid (positive here, invertible in general).

---

## 5. A gallery of $2 \times 2$ generators

Before going to higher dimensions, it helps to see that different generators produce visibly different motions. Each row below is a $2 \times 2$ matrix $M$ that you can exponentiate by hand.

| generator $M$ | velocity field $\dot z = Mz$ | $\exp(tM)$ | motion |
|---|---|---|---|
| $\begin{pmatrix}0&-1\\1&0\end{pmatrix} = J$ | perpendicular to $z$ | $R(t)$ | pure rotation |
| $\begin{pmatrix}\lambda_1&0\\0&\lambda_2\end{pmatrix}$ | along the axes | $\begin{pmatrix}e^{\lambda_1 t}&0\\0&e^{\lambda_2 t}\end{pmatrix}$ | stretch or squash each axis |
| $\begin{pmatrix}0&1\\0&0\end{pmatrix} = N$ | horizontal, proportional to height | $\begin{pmatrix}1&t\\0&1\end{pmatrix}$ | shear (like sliding a deck of cards) |
| $aI + bJ$ | rotate plus grow or shrink | $e^{at} R(bt)$ | spiral in ($a<0$) or out ($a>0$) |

Here is how to derive two of these rows:

**The shear.** $N^2 = 0$, so every term of the series after the second vanishes and $\exp(tN) = I + tN$. A matrix with $N^k = 0$ for some $k$ is called *nilpotent*, and its exponential is a finite polynomial.

**The spiral.** The identity matrix commutes with everything, so $aI$ and $bJ$ commute. By §4.4's rule for commuting generators, $\exp(aI + bJ) = \exp(aI)\exp(bJ) = e^{a} R(b)$. The eigenvalues of $M = aI + bJ$ are $a \pm bi$: the real part $a$ sets the growth rate $e^{a}$ and the imaginary part $b$ sets the angle of rotation. This is the "readable spectrum" from Chapter 3 §4, verified on the smallest possible case.

> **Takeaway.** A $2 \times 2$ generator has four free numbers and lives in a flat, four-dimensional space $\mathfrak{gl}(2)$. Every choice gives a valid invertible operator. The kind of motion (rotation, stretch, shear, spiral) can be read from the generator's eigenvalues.

---

## 6. Rotations in 3-D: generators that don't commute

Everything so far has been commutative. In two dimensions all rotations share a single centre, so the order of turning doesn't matter. The interesting part of the theory begins in three dimensions.

### 6.1 The worked example: turning a book

Put a book flat on a desk. Then do two experiments.
- **Experiment A.** Turn it 90° about the $x$-axis, then 90° about the $y$-axis.
- **Experiment B.** Do the same two turns in the opposite order.

The book ends up in different orientations. (Try it with a phone. The check script confirms that the two final rotation matrices differ.) **In 3-D, rotations do not commute.**

### 6.2 The three generators of $SO(3)$

Turning about each coordinate axis is a rotation in one coordinate plane, so each has a generator like $J$, embedded in a $3 \times 3$ matrix:

$$L_x = \begin{pmatrix} 0&0&0\\0&0&-1\\0&1&0 \end{pmatrix},\qquad L_y = \begin{pmatrix} 0&0&1\\0&0&0\\-1&0&0 \end{pmatrix},\qquad L_z = \begin{pmatrix} 0&-1&0\\1&0&0\\0&0&0 \end{pmatrix}.$$

- $L_z$ is $J$ placed in the $(x, y)$ block: it spins the $xy$-plane and leaves $z$ fixed.
- $L_x$ and $L_y$ do the same for the $yz$- and $zx$-planes.

All three are **skew-symmetric**: $L^\top = -L$. The Lie algebra $\mathfrak{so}(3)$ is the set of all real combinations $\omega_x L_x + \omega_y L_y + \omega_z L_z$, which is exactly the set of $3 \times 3$ skew-symmetric matrices. It is a flat, three-dimensional space, one dimension per axis of rotation.

### 6.3 The commutator: an exact measure of failure to commute

To measure how much two generators fail to commute, define the **commutator**, also called the **Lie bracket**:

$$[X, Y] = XY - YX.$$

It is zero exactly when $X$ and $Y$ commute.

**Worked computation of $[L_x, L_y]$.** First the two products. For $L_x L_y$, row $i$ of $L_x$ times column $j$ of $L_y$:

$$L_x L_y = \begin{pmatrix} 0&0&0\\0&0&-1\\0&1&0 \end{pmatrix}\begin{pmatrix} 0&0&1\\0&0&0\\-1&0&0 \end{pmatrix} = \begin{pmatrix} 0&0&0\\1&0&0\\0&0&0 \end{pmatrix}.$$

(The only nonzero entry is row 2 · column 1: $(0)(0) + (0)(0) + (-1)(-1) = 1$.)

$$L_y L_x = \begin{pmatrix} 0&0&1\\0&0&0\\-1&0&0 \end{pmatrix}\begin{pmatrix} 0&0&0\\0&0&-1\\0&1&0 \end{pmatrix} = \begin{pmatrix} 0&1&0\\0&0&0\\0&0&0 \end{pmatrix}.$$

(The only nonzero entry is row 1 · column 2: $(1)(1) = 1$.)

Subtracting:

$$[L_x, L_y] = \begin{pmatrix} 0&-1&0\\1&0&0\\0&0&0 \end{pmatrix} = L_z .$$

By the same calculation, $[L_y, L_z] = L_x$ and $[L_z, L_x] = L_y$, a cyclic pattern. **The bracket of two generators is again a generator**, and here it is the third rotation axis.

### 6.4 Why the bracket is *the* measure: the infinitesimal round trip

Here is the most important calculation in this chapter. It shows why the bracket, rather than some other combination, is the right object.

Perform four small motions in a loop, each for a short time $\varepsilon$:
1. move along $X$,
2. then along $Y$,
3. then back along $-X$,
4. then back along $-Y$.

If $X$ and $Y$ commuted, you would end up exactly where you started. The net group element of the loop is

$$g = e^{\varepsilon X} e^{\varepsilon Y} e^{-\varepsilon X} e^{-\varepsilon Y}.$$

(This is called the *group commutator*.) Expand each exponential to second order in $\varepsilon$, using $e^{\varepsilon X} \approx I + \varepsilon X + \tfrac{\varepsilon^2}{2}X^2$, and so on.

**Step 1: the first two factors.** Multiply them and keep terms up to $\varepsilon^2$:
$$e^{\varepsilon X}e^{\varepsilon Y} \approx I + \varepsilon(X + Y) + \varepsilon^2\Big(\tfrac12 X^2 + XY + \tfrac12 Y^2\Big).$$

**Step 2: the last two factors.** These are the same with $\varepsilon \to -\varepsilon$. The $\varepsilon^2$ terms don't change sign:
$$e^{-\varepsilon X}e^{-\varepsilon Y} \approx I - \varepsilon(X + Y) + \varepsilon^2\Big(\tfrac12 X^2 + XY + \tfrac12 Y^2\Big).$$

**Step 3: multiply the two results.** Write $S = X + Y$ and $Q = \tfrac12 X^2 + XY + \tfrac12 Y^2$, so the factors are $(I + \varepsilon S + \varepsilon^2 Q)(I - \varepsilon S + \varepsilon^2 Q)$. Collect by order:
- order $\varepsilon^0$: $I$;
- order $\varepsilon^1$: $S - S = 0$, so the first-order terms cancel;
- order $\varepsilon^2$: $Q + Q - S^2$. The $-S^2$ is the product of the two first-order terms, $(\varepsilon S)(-\varepsilon S)$.

**Step 4: simplify the second-order term.** Expand $S^2 = (X+Y)^2 = X^2 + XY + YX + Y^2$. Note that $XY$ and $YX$ stay separate because order matters. Then
$$2Q - S^2 = (X^2 + 2XY + Y^2) - (X^2 + XY + YX + Y^2) = XY - YX.$$

**Result:**
$$\boxed{\enspace e^{\varepsilon X}e^{\varepsilon Y}e^{-\varepsilon X}e^{-\varepsilon Y} = I + \varepsilon^2[X, Y] + O(\varepsilon^3).\enspace}$$

In words: **the small loop fails to close, and the gap is the bracket, scaled by $\varepsilon^2$.** The bracket is the direction you drift in when you try to go out and back along two non-commuting motions. The check script measures this. As $\varepsilon$ shrinks tenfold, the error in the boxed formula also shrinks tenfold, as the $O(\varepsilon^3)$ remainder predicts.

For the book: wiggle a little about $x$, a little about $y$, then undo each. The book ends up turned slightly about $z$, because $[L_x, L_y] = L_z$.

### 6.5 A familiar disguise: the cross product

Write a skew-symmetric $3\times 3$ matrix as $\hat v = v_x L_x + v_y L_y + v_z L_z$ for a vector $v = (v_x, v_y, v_z)$. This is called the *hat map*. Two identities, both checked in the script, show you have seen $\mathfrak{so}(3)$ before:

$$\hat v w = v \times w, \qquad [\hat v, \hat w] = \widehat{v \times w}.$$

**$\mathfrak{so}(3)$ is $\mathbb{R}^3$ with the cross product as its bracket.** Angular velocity in physics is a Lie-algebra element, and the cross product is the bracket.

---

## 7. Naming what we just saw: a Lie algebra

We can now state the abstraction, since every part of it has appeared in an example.

> **Definition.** A **Lie algebra** is a vector space $\mathfrak{g}$ with a bracket operation $[\cdot,\cdot] : \mathfrak{g} \times \mathfrak{g} \to \mathfrak{g}$ satisfying three rules:
> 1. **Bilinearity.** $[aX + bY, Z] = a[X,Z] + b[Y,Z]$, and the same in the second slot. (The bracket respects addition and scaling.)
> 2. **Antisymmetry.** $[X, Y] = -[Y, X]$. In particular, $[X, X] = 0$: every motion commutes with itself.
> 3. **The Jacobi identity.** $[X,[Y,Z]] + [Y,[Z,X]] + [Z,[X,Y]] = 0$.

Here is how to read each rule:

- **Rule 1** says the bracket is compatible with the flat structure, so you can compute it one basis element at a time.
- **Rule 2** is visible in the round trip of §6.4. Running the loop in the opposite order ($Y$ first) gives a drift of $\varepsilon^2[Y,X]$, which points the opposite way. **Swapping the order flips the sign of the bracket.** This is the "swap-odd" property behind Chapter 5 §6: a simultaneous double knockdown has no first and second gene, so it cannot see the sign of the bracket.
- **Rule 3** looks like bookkeeping, but it is what makes the bracket consistent with associativity in the group. For matrices it holds automatically: expand all twelve products and they cancel in pairs. (The script checks this on random $4 \times 4$ matrices.)

**The phrase "with the bracket" carries real weight, through closure.** The bracket of two elements must land back in $\mathfrak{g}$. That is a real constraint on which sets of matrices count as Lie algebras. Here are two cases worked out:

- **Skew-symmetric matrices are closed.** If $K_1^\top = -K_1$ and $K_2^\top = -K_2$, then
  $$[K_1,K_2]^\top = (K_1K_2 - K_2K_1)^\top = K_2^\top K_1^\top - K_1^\top K_2^\top = K_2K_1 - K_1K_2 = -[K_1,K_2].$$
  The bracket is skew again, so $\mathfrak{so}(D)$ is a Lie algebra.
- **Symmetric matrices are not closed.** Take $S_1 = \begin{pmatrix}1&0\\0&-1\end{pmatrix}$ (stretch $x$, squash $y$) and $S_2 = \begin{pmatrix}0&1\\1&0\end{pmatrix}$ (stretch along the diagonal). Then
  $$[S_1, S_2] = \begin{pmatrix}0&2\\-2&0\end{pmatrix} = -2J,$$
  which is a *rotation* generator. **The bracket of two stretches along different axes is a rotation.** Correspondingly, $\exp(S_1)\exp(S_2)$ is not symmetric (the script confirms this): composing two pure stretches gives a stretch combined with a rotation.

> **A presentation subtlety this exposes.** Chapter 3 §5's table says "symmetric $M$ → positive-definite $\exp(M)$, a pure stretch". That is true for one operator. But the pure stretches don't form a subgroup: composing two of them produces a rotation. Constraining the generator to the skew-symmetric matrices gives a closed system (a subalgebra, and its group $SO(D)$). Constraining it to the symmetric matrices gives a well-behaved single operator, but the closure is lost as soon as two interventions are composed.

### 7.1 Structure constants: a Lie algebra in a table

Choose a basis $B_1, \dots, B_k$ for a Lie algebra. By bilinearity, the bracket is determined by the brackets of basis pairs. By closure, each such bracket is again a combination of basis elements:

$$[B_i, B_j] = \sum_{\ell=1}^{k} c_{ij}^{\ell} B_\ell .$$

The numbers $c_{ij}^{\ell}$ are the **structure constants**. Here $i, j$ label the two inputs and $\ell$ labels the output. They are a complete "multiplication table" for the algebra. For $\mathfrak{so}(3)$ with basis $(L_x, L_y, L_z)$, the table is $[L_x, L_y] = L_z$ and its cyclic versions, so $c_{xy}^{z} = 1$ and so on.

This is exactly the object Chapter 4 means when it says the basis "confines every bracket the model can produce". If the span of $\{B_i\}$ is closed, every interaction the model can express lies in $\mathrm{span}\{[B_i, B_j]\}$, which is a small, named, inspectable set of directions.

---

## 8. More Lie algebras, each with a story

The goal of this section is fluency: one memorable story per algebra, which you can use when presenting the subject.

### 8.1 The Heisenberg algebra: where BCH stops after one bracket

Take $3 \times 3$ matrices with zeros on and below the diagonal (*strictly upper-triangular*). A basis is given by the matrix units $E_{ij}$, which have a single 1 in row $i$, column $j$ and zeros elsewhere:

$$X = E_{12},\quad Y = E_{23},\quad Z = E_{13}.$$

The brackets are:
- $XY = E_{12}E_{23} = E_{13}$, while $YX = E_{23}E_{12} = 0$. So $[X, Y] = Z$.
- $[X, Z] = 0$ and $[Y, Z] = 0$. Both products are zero in each order. So $Z$ commutes with everything; it is *central*.

The algebra therefore produces one new direction ($Z$) and then stops. Its consequence for composition is clean: the BCH formula of §9 ends after a single bracket, *exactly*, with no approximation:

$$e^X e^Y = e^{X + Y + \frac12 [X,Y]}.$$

**Check by hand.**
- $X^2 = Y^2 = 0$, so $e^X = I + X$ and $e^Y = I + Y$. Their product is $(I+X)(I+Y) = I + E_{12} + E_{23} + E_{13}$.
- On the other side, let $W = E_{12} + E_{23} + \tfrac12 E_{13}$. Then $W^2 = E_{13}$ (only $E_{12}E_{23}$ survives) and $W^3 = 0$. So $e^{W} = I + W + \tfrac12 W^2 = I + E_{12} + E_{23} + \tfrac12E_{13} + \tfrac12 E_{13}$. The two sides match.
- Leaving out the bracket term gives an error of exactly $0.5$ in the corner entry.

**The story.** This is the algebra of quantum mechanics: position and momentum satisfy $[\hat x, \hat p] = i\hbar$, which is a central element. It is also the algebra of any two operations whose failure to commute is a fixed offset, independent of the state. **It is the ideal case for the bracket-predicts-interaction hypothesis**: the interaction is determined entirely by the single bracket, at any scale, with no radius of validity to worry about.

### 8.2 $\mathfrak{se}(2)$: parallel parking

A car on a flat lot has a position $(x, y)$ and a heading. Its rigid motions form $SE(2)$, the *special Euclidean group*. To write them as matrices, use $3 \times 3$ matrices acting on $(x, y, 1)^\top$ (*homogeneous coordinates*), where the third column carries translations. Two generators:

$$R = \begin{pmatrix}0&-1&0\\1&0&0\\0&0&0\end{pmatrix}\ \text{(turn in place)},\qquad T = \begin{pmatrix}0&0&1\\0&0&0\\0&0&0\end{pmatrix}\ \text{(drive forward along } x\text{)}.$$

Compute the bracket:
- $RT$: the only nonzero entry is row 2 · column 3, which is $1$. So $RT = E_{23}$.
- $TR$: row 1 of $T$ selects row 3 of $R$, which is zero. So $TR = 0$.

$$[R, T] = E_{23} = \text{slide sideways along } y.$$

**The story.** A car cannot move sideways directly, since it has no "slide" control. But by §6.4, the manoeuvre "turn a little, drive a little, turn back, reverse a little" produces a net displacement of $\varepsilon^2 [R, T]$, which is sideways motion. This is how parallel parking works. The check script measures the manoeuvre's displacement divided by $\varepsilon^2$ and gets $(\approx 0, 1)$.

**The general lesson.** Brackets can produce *new directions* that none of the original generators contains. The original controls span a 2-D set, and their bracket adds a third direction. (In control theory this is the Chow–Rashevskii theorem: you can reach every direction in the span of the controls and all their iterated brackets.) For the action operator, this is the hopeful reading of epistasis: two knockdowns, each moving the latent state in its own way, combine to produce a shift neither causes alone, in a direction given by their bracket.

### 8.3 $\mathfrak{sl}(2)$: trace zero, area preserved

Take the $2\times 2$ matrices with trace zero (the *trace* is the sum of the diagonal entries). A standard basis:

$$H = \begin{pmatrix}1&0\\0&-1\end{pmatrix},\quad E = \begin{pmatrix}0&1\\0&0\end{pmatrix},\quad F = \begin{pmatrix}0&0\\1&0\end{pmatrix},$$

with brackets $[H, E] = 2E$, $[H, F] = -2F$ and $[E, F] = H$.

**Why trace zero?** For any square matrix,

$$\det\big(\exp(M)\big) = e^{\mathrm{tr}(M)}.$$

The reason: the determinant is the product of the eigenvalues, the trace is their sum, and $\exp$ sends each eigenvalue $\lambda$ to $e^\lambda$, so a product of exponentials becomes the exponential of a sum. A trace-zero generator therefore gives $\det = e^0 = 1$, an operator that preserves area.

**The story.** The trace of the generator is the rate at which volume grows. This gives a cheap, readable diagnostic for a learned operator. $\mathrm{tr}(M_g)$ says whether intervention $g$ expands or contracts the latent cloud overall, before looking at any direction in detail. A consequence: $\det e^M > 0$ always, so $\exp$ can never produce an orientation-reversing operator, and the network can only reach $GL^+(D)$.

### 8.4 The abelian algebras: when everything is additive

Take the diagonal matrices. Any two of them commute, so every bracket is zero. A Lie algebra in which every bracket vanishes is called **abelian**. In an abelian algebra, $e^X e^Y = e^{X+Y}$ always, and composition is exactly addition.

**The story.** This is the algebra of the round-7 finding: the training-free $\Delta(A) + \Delta(B)$ prediction scores $\Delta r = 0.961$. To first approximation, the data behave as if the relevant generators lived in an abelian algebra.

### 8.5 $\mathfrak{gl}(D)$: everything, and why that is a problem

The algebra of $GL(D)$ is all $D \times D$ matrices, written $\mathfrak{gl}(D)$, with the commutator as bracket. It is trivially closed, since any product of matrices is a matrix. It has dimension $D^2$, which is 65,536 for $D = 256$.

**The story.** This is Round 4's `NamedGeneratorOperator`: one free generator per gene in $\mathfrak{gl}(256)$. The bracket of two such generators can point anywhere in a 65,536-dimensional space. Most of that space was never constrained by the training loss, which fixed each operator's effect only on the directions the data occupy. That is the identifiability diagnosis of Chapter 7, restated in the language of this chapter: in $\mathfrak{gl}(D)$ the bracket is not *confined*, so it is mostly a measure of size. A shared low-rank basis replaces $\mathfrak{gl}(D)$ with a small span and a small table of structure constants.

### 8.6 Beyond matrices: vector fields

One example lies outside matrices, for completeness. The smooth vector fields on a space form a Lie algebra under the vector-field bracket $[u, v] = (u\cdot\nabla)v - (v\cdot\nabla)u$. Matrix generators are the special case of *linear* vector fields $z \mapsto Mz$. Applied to $u = Xz$ and $v = Yz$, the formula gives $[u, v](z) = (YX - XY)z$: the matrix commutator with the opposite sign. This sign difference is a known convention clash between the two settings. The general version is the language of nonlinear flows, and the bridge back to the Koopman picture of Chapter 2, which trades a nonlinear vector field on states for a linear operator on functions of states.

---

## 9. Composition in general: the Baker–Campbell–Hausdorff formula

### 9.1 The question

The question is: given two generators $X$ and $Y$, what single generator $Z$ satisfies $e^X e^Y = e^Z$? If $X$ and $Y$ commute, $Z = X + Y$ (§4.4). In general, $Z$ is $X + Y$ plus corrections.

### 9.2 The worked derivation to second order

Treat $X$ and $Y$ as small, and keep terms up to second order (products of two generators).

**Step 1: the left side.**
$$e^X e^Y \approx \big(I + X + \tfrac12X^2\big)\big(I + Y + \tfrac12 Y^2\big) \approx I + X + Y + \tfrac12 X^2 + XY + \tfrac12 Y^2.$$

**Step 2: guess the form of $Z$.** Let $Z = X + Y + W$, where $W$ is a second-order correction to be found. Then
$$e^Z \approx I + Z + \tfrac12 Z^2 \approx I + X + Y + W + \tfrac12 (X^2 + XY + YX + Y^2).$$
(In $Z^2$, any term involving $W$ is third order or higher, so it is dropped.)

**Step 3: match the second-order terms.** For the two sides to agree, $W + \tfrac12(XY + YX) = XY$, so

$$W = \tfrac12 (XY - YX) = \tfrac12 [X, Y].$$

**Result: the start of the Baker–Campbell–Hausdorff (BCH) formula:**

$$\log\big(e^X e^Y\big) = X + Y + \tfrac12 [X, Y] + \tfrac{1}{12}\big([X,[X,Y]] + [Y,[Y,X]]\big) + \cdots$$

Here $\log$ is the inverse of $\exp$ near the identity: it takes an operator back to its generator.

Here is how to read the formula. The first-order part, $X + Y$, is additivity. Every correction beyond it is built from **brackets** and **nested brackets**: $[X, Y]$ at second order, brackets of brackets at third order, and so on. No plain product like $XY$ appears on its own. That is the formal sense in which the bracket "is" the interaction.

> **Why only brackets.** This is the precise statement, which [Chapter 5 §3](05-composition.md) phrases loosely. BCH is an identity that holds in *every* Lie algebra, including abstract ones in which the bracket is the only product defined (for example, $\mathbb{R}^3$ with the cross product, §6.5). A universal formula can only use operations every Lie algebra has, and those are addition, scaling and the bracket. Whether the set of matrices you are working in happens to be closed under ordinary multiplication is beside the point. $\mathfrak{gl}(D)$ is closed under multiplication, and BCH still contains only brackets.

### 9.3 The symmetric product, and what it cancels

Chapter 5 composes two simultaneous knockdowns as $e^{X/2} e^Y e^{X/2}$, with half of $X$ on each side. Apply the second-order result twice:

**Step 1.**
$$\log\big(e^{X/2}e^{Y}\big) \approx \tfrac{X}{2} + Y + \tfrac12\big[\tfrac{X}{2}, Y\big] = \tfrac{X}{2} + Y + \tfrac14[X,Y].$$

**Step 2.** Compose that with the remaining $e^{X/2}$:
$$\Big(\tfrac{X}{2} + Y + \tfrac14 [X,Y]\Big) + \tfrac{X}{2} + \tfrac12\Big[\tfrac{X}{2} + Y,\ \tfrac{X}{2}\Big].$$
In the last bracket, $[\tfrac X2, \tfrac X2] = 0$ by antisymmetry, so it equals $\tfrac12[Y, \tfrac X2] = -\tfrac14 [X, Y]$.

**Step 3.** The two $\tfrac14[X,Y]$ terms cancel:
$$\log\big(e^{X/2}e^{Y}e^{X/2}\big) = X + Y + (\text{third order}).$$

The sign-flipping second-order term is gone. The script also measures the leading third-order remainder, $-\tfrac1{24}[X,[X,Y]] - \tfrac1{12}[Y,[X,Y]]$. Subtracting it reduces the error to fifth order.

> **Precision note for when you present this.** The surviving double-commutator term in the symmetric product is not fully swap-invariant: exchanging $X$ and $Y$ moves the $\tfrac1{24}$ and $\tfrac1{12}$ coefficients between the two nested brackets. What is exactly swap-invariant is the *even part*, $\tfrac12\big(\log(e^Xe^Y) + \log(e^Ye^X)\big)$, which [`checks/verify_bch.py`](checks/verify_bch.py) §2 tests. The qualitative claim stands: a simultaneous experiment sees only the even, magnitude-like part of non-commutativity, never the sign of $[X,Y]$.

### 9.4 The radius of validity, in one sentence

BCH is a power series in the size of the generators, so it is only trustworthy when $\lVert X \rVert$ and $\lVert Y \rVert$ are small, roughly below 1 (a standard sufficient condition for convergence is $\lVert X\rVert + \lVert Y\rVert < \ln 2$). At $\lVert M \rVert \approx 12$, as in Round 4's weak-prior arm, the generators are far outside that range, the truncated series is no longer a good approximation, and the bracket stops meaning "the interaction". This is the radius [Chapter 5 §4](05-composition.md) measures.

---

## 10. Two cautions about $\exp$ worth knowing cold

**$\exp$ is not one-to-one.** $\exp(2\pi J) = R(2\pi) = I = \exp(0)$. Two different generators, $0$ and $2\pi J$, produce the same operator, so "the generator of an operator" is not unique. Any rotational part can be wound an extra full turn. For the action operator this means a learned $M$ is pinned down only up to such choices, and brackets computed from two equivalent generators can differ. This is another route to the identifiability problem, separate from the unconstrained directions of §8.5.

**$\exp$ does not reach every invertible matrix.**
- Its outputs always have $\det > 0$ (§8.3), so no reflection is reachable.
- Even some matrices with positive determinant are unreachable. For example, $\begin{pmatrix}-1&1\\0&-1\end{pmatrix}$ has $\det = 1$ but no real logarithm. (A real matrix has a real logarithm only if each Jordan block with a negative eigenvalue appears an even number of times.)

The network can therefore represent every operator that is the time-one map of some linear flow, and nothing else. Chapter 3 §3 called this "the assertion that a flow produced $A$". These are the operators that assertion excludes.

---

## 11. Back to the action operator: a translation table

| this chapter | the action-operator series | the code |
|---|---|---|
| Lie group $G$ | the operator space, $GL^+(256)$ or a subgroup | the set of possible `A` |
| Lie algebra $\mathfrak{g}$ | where the network's output lives | the set of possible `M` |
| generator, velocity field $\dot z = Mz$ | an intervention's instantaneous tendency | `M = sum(alpha_i * B_i)` |
| $\exp: \mathfrak{g}\to G$ | time-one map of the flow | `torch.matrix_exp(M)` |
| identity $I = \exp(0)$ | "intervention does nothing" | zero-initialized last layer |
| subalgebra (e.g. skew) → subgroup | structure by construction | `SkewBasis`, `SE3Basis` |
| basis $\{B_i\}$, structure constants $c_{ij}^\ell$ | what is representable, and where brackets can point | `FreeBasis`, `NamedBasis`, … |
| $\mathfrak{gl}(D)$, unconfined brackets | Round 4, one dense generator per gene | `NamedGeneratorOperator` |
| bracket $[X,Y]$ | predicted interaction (epistasis) | `‖[M_A, M_B]‖` vs `rel_GI` |
| BCH and its radius | when the bracket can be trusted | `checks/verify_bch_radius.py` |
| symmetric product $e^{X/2}e^Ye^{X/2}$ | simultaneous double knockdown | the composition used in Round 4 |
| abelian algebra | pure additivity, $\Delta(A)+\Delta(B)$ | the round-7 baseline, $\Delta r = 0.961$ |

---

## 12. Presenting it: three lengths

**The one-sentence version.** "Interventions are flows: the network outputs a velocity field $M$ in a flat space (the Lie algebra), the matrix exponential turns it into a guaranteed-valid transformation (a Lie group element), and how two interventions interact is measured by how their velocity fields fail to commute, $[M_A, M_B] = M_AM_B - M_BM_A$."

**The two-minute version.** Use the book-turning demonstration (§6.1) to show that order matters. Then the clock hand (§3–4) to show generator versus group element, with $\exp(\theta J) = R(\theta)$ derived on the board from $J^2 = -I$. Then parallel parking (§8.2) to show that the bracket is a *new direction* you reach by a small out-and-back loop. Close with BCH: "composition is addition plus brackets."

**The questions you will get, and short answers.**
- *"Why not just learn $A$?"* Because the invertible matrices are not a flat space: sums can be singular (§1.3). $\exp$ gives invertibility, an identity at zero, and clean composition for free.
- *"Is $\exp$ just a trick?"* No. It is the solution of $\dot z = Mz$. Writing $A = \exp(M)$ asserts that a flow produced $A$ (§4.2).
- *"Why the commutator and not some other combination?"* It is the leading term of the out-and-back loop (§6.4) and the only correction BCH ever uses (§9.2).
- *"Can the model represent any linear map?"* No. It is restricted to $\det > 0$, and even then not every such map is reachable (§10). The restriction is what makes the model interpretable.
- *"Why did it fail on Perturb-seq?"* The generators lived in all of $\mathfrak{gl}(256)$, so the brackets were not confined and mostly measured size. Where BCH is valid, the interactions are small. And a simultaneous experiment cannot see the sign of the bracket (§7, §9.3).

---

## 13. Exercises, with answers

1. **Compute $\exp\begin{pmatrix}0&t\\0&0\end{pmatrix}$.**
   *Answer:* $\begin{pmatrix}1&t\\0&1\end{pmatrix}$, because the matrix squares to zero and the series stops after two terms.

2. **Show $[\hat v,\hat w] = \widehat{v\times w}$ for $v = e_x$, $w = e_y$.**
   *Answer:* $\hat e_x = L_x$, $\hat e_y = L_y$, and $[L_x,L_y] = L_z = \hat e_z = \widehat{e_x \times e_y}$.

3. **Are the upper-triangular $2\times2$ matrices closed under the bracket? What is $[\mathrm{diag}(1,0),E_{12}]$?**
   *Answer:* Yes, they are closed. The bracket is $E_{12}$, so it stays upper-triangular and in fact strictly upper-triangular.

4. **If $\mathrm{tr} M_g = -3$, what does intervention $g$ do to the volume of the latent cloud?**
   *Answer:* It multiplies volume by $e^{-3} \approx 0.05$, a strong contraction.

5. **Two generators satisfy $[X,Y] = 0.01I$. How far is $e^Xe^Y$ from $e^{X+Y}$, and does the radius of validity matter?**
   *Answer:* This is Heisenberg-type, since $I$ is central. So $e^Xe^Y = e^{X+Y}e^{0.005 I} = 1.005 e^{X+Y}$ exactly. The radius of validity does not matter, because there are no higher terms.

6. **Why does the network's zero-initialized last layer start every operator at the identity, while zero-initializing a direct-$A$ model does not?**
   *Answer:* $\exp(0) = I$, the no-op. A direct model with $A = 0$ is the annihilation map, which sends every state to zero.
