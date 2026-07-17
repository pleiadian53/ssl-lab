# The Algebra of Composition

*What happens when you apply two operators instead of one. The [operator gallery](02-operator-gallery.md) ended by noting that order matters, and that the gap is "governed by the commutator." This note makes good on that sentence: what the commutator is, why it is the exact measure of non-additivity, why a formula called BCH can only ever produce commutators, and why its symmetric twin, the anticommutator, is a different animal that answers a different question.*

> **Prerequisites.** The [gallery](02-operator-gallery.md) is enough: an operator is a matrix $A$ acting on a latent vector $z$, built as $A = \exp(M)$ from a **generator** $M$, where $\exp(M) = I + M + \tfrac{1}{2}M^2 + \tfrac{1}{6}M^3 + \cdots$ uses real matrix powers. Nothing else is assumed. Every symbol is defined at first use, and the only prior fact we lean on is that matrix multiplication does not generally commute.

---

## 1. Composing two operators, and a notational trap

One operator transforms a state: $z' = A z$. Two operators applied in sequence transform it twice. Apply $A$ first, then $B$:

$$z'' = B(Az) = (BA) z.$$

So **composition is matrix multiplication**, and here is the trap worth absorbing immediately: "apply $A$ first, then $B$" is written $BA$, **right to left**. The operator applied first sits on the right, nearest the vector it touches first. This ordering convention bites everyone once, and it is the reason a time-ordered product in a world model runs backwards on the page relative to the clock.

Now the question that generates everything below. Does the order matter? That is: does $AB$ equal $BA$?

For ordinary numbers, always. $3 \times 5 = 5 \times 3$. For matrices, **almost never**. Take a rotation $R$ and a stretch $S$ in the plane: rotate-then-stretch lands somewhere different from stretch-then-rotate, because the stretch acts along fixed axes and the rotation moves the state relative to those axes. Multiplication of matrices is the mathematics of "doing things in sequence," and sequence is exactly where order matters.

We say $A$ and $B$ **commute** when $AB = BA$, and that they **fail to commute** otherwise.

## 2. The commutator: a number-free measure of "did the order matter?"

Instead of asking the yes/no question, measure the gap. The **commutator** of $A$ and $B$ is

$$[A, B] = AB - BA.$$

Read it as *the difference between doing it one way and doing it the other*. It is a matrix, not a number, and it has one defining property:

$$[A, B] = 0 \quad \Longleftrightarrow \quad AB = BA \quad \Longleftrightarrow \quad A \text{ and } B \text{ commute}.$$

The commutator is zero exactly when order is irrelevant, and the "larger" it is, the more the order matters. That is the whole idea, and everything else in this note is a consequence.

Two facts to keep. First, $[A, A] = AA - AA = 0$: **anything commutes with itself**, which is obvious and will matter a great deal in §7. Second, swapping the arguments flips the sign:

$$[B, A] = BA - AB = -(AB - BA) = -[A, B].$$

## 3. When does $\exp(A)\exp(B) = \exp(A+B)$?

Now bring in the exponential, because that is how the gallery builds operators. If generators simply added, life would be easy: applying the operator of $A$ and then the operator of $B$ would be the operator of the summed generator $A + B$. Sometimes that is true, and there is an exact condition for it.

**The theorem.** If $[A,B] = 0$, then

$$\exp(A)\exp(B) = \exp(A+B).$$

**Why.** Multiply the two series and collect terms of the same total degree $n$:

$$\exp(A)\exp(B) = \left(\sum_i \frac{A^i}{i!}\right)\left(\sum_j \frac{B^j}{j!}\right) = \sum_n \frac{1}{n!} \sum_{i+j=n} \frac{n!}{i!\ j!} A^i B^j.$$

That inner sum is the **binomial expansion** of $(A+B)^n$, and it collects into $(A+B)^n$ only if you are allowed to reorder $A$'s past $B$'s while gathering terms. For $n=2$ the point is visible with no bookkeeping:

$$(A+B)^2 = A^2 + AB + BA + B^2,$$

which equals $A^2 + 2AB + B^2$ **only when** $AB = BA$. So the binomial theorem, and with it the whole identity, rests on commutativity and on nothing else.

**The converse, honestly.** The reverse implication needs a caveat that is usually glossed over. It is *not* true in complete generality that $\exp(A)\exp(B) = \exp(A+B)$ forces $[A,B]=0$; there are exotic counterexamples built from eigenvalues that differ by integer multiples of $2\pi i$, where the exponential wraps around and coincidences occur. But those live far from the identity. **Near the identity**, where $M$ is small, $\log$ is well defined and the correspondence between generators and operators is one-to-one, the equivalence does hold:

$$[A, B] = 0 \quad \Longleftrightarrow \quad \exp(A)\exp(B) = \exp(A+B) \qquad \text{(for small } A, B\text{)}.$$

This matters because it is exactly the regime our operators are built to live in. A zero-initialized generator starts at $M = 0$, and a least-action penalty on $\lVert M \rVert$ keeps it near there. The near-identity prior is not only an inductive bias about the world; it is also what makes the algebra behave.

> **The equivalence to carry forward.** For near-identity operators: *generators commute* $\iff$ *composition is additive*. Non-commutativity **is** the departure from additivity. That is not an analogy. It is what the group product computes.

## 4. Baker–Campbell–Hausdorff: what composition costs when they don't commute

If $[A,B] \ne 0$, then $\exp(A)\exp(B)$ is still *some* operator, and near the identity it is still the exponential of *some* generator. Which one? The answer is the **Baker–Campbell–Hausdorff formula** (BCH):

$$\log\big(\exp(A)\exp(B)\big) = A + B + \tfrac{1}{2}[A,B] + \tfrac{1}{12}\big([A,[A,B]] + [B,[B,A]]\big) - \tfrac{1}{24}\big[B,[A,[A,B]]\big] + \cdots$$

Read it as a **correction series**. The generator of the composed operator is the naive sum $A + B$, plus an infinite tail of corrections. Every correction is built from commutators, so:

- if $[A,B] = 0$, **every single correction term vanishes** (they all contain a bracket of $A$ and $B$ somewhere inside), and the formula collapses to $A + B$, recovering §3;
- if $[A,B] \ne 0$, the leading correction is $\tfrac{1}{2}[A,B]$, and the rest are smaller nested brackets.

You can see the first correction with the same degree-2 bookkeeping as before. Expanding to second order,

$$\exp(A)\exp(B) = I + A + B + \tfrac{1}{2}A^2 + AB + \tfrac{1}{2}B^2 + \cdots$$
$$\exp(A+B) = I + A + B + \tfrac{1}{2}\big(A^2 + AB + BA + B^2\big) + \cdots$$

Subtract: the difference at this order is $AB - \tfrac{1}{2}(AB + BA) = \tfrac{1}{2}(AB - BA) = \tfrac{1}{2}[A,B]$. The commutator is not inserted into BCH by decree. It is what falls out when you ask how far the product is from the sum.

## 5. Why BCH contains *only* nested commutators

This is the structural fact that decides which objects are allowed to appear in an operator algebra, and it has a satisfying reason.

The theorem (Baker, Campbell, Hausdorff, sharpened by Dynkin) says every term of the series is a rational multiple of an **iterated commutator** of $A$ and $B$. Not $AB$. Not $A^2$. Not $AB + BA$. Only brackets of brackets.

**Why it must be so.** Generators are not arbitrary matrices; they live in a **Lie algebra**, a vector space that is closed under the commutator but *not* under plain matrix multiplication. The logarithm of a group element must land back in that algebra, so the only operations BCH is permitted to use are the ones the algebra is closed under. It can add, it can scale, it can bracket. It cannot multiply.

**A concrete case that makes this vivid.** Take the rotations, whose generators are the **skew-symmetric** matrices ($M^\top = -M$, the family the gallery identifies with pure rotation). Let $A$ and $B$ both be skew. Is their commutator still skew?

$$[A,B]^\top = (AB - BA)^\top = B^\top A^\top - A^\top B^\top = (-B)(-A) - (-A)(-B) = BA - AB = -[A,B].$$

Yes. The bracket of two rotation generators is a rotation generator; the algebra is closed. Now try the product:

$$(AB)^\top = B^\top A^\top = (-B)(-A) = BA \ne -AB \quad\text{in general.}$$

The plain product of two rotation generators is **not** a rotation generator. It has fallen out of the algebra and no longer describes a rotation at all.

So the restriction is not fussiness. Multiplication *destroys the structure* that made the generator meaningful, while the commutator preserves it. That is why the bracket, and only the bracket, is the algebra's native operation, and why every composition correction in the universe is made of brackets.

## 6. The odd and even halves of a product

Here is a decomposition that puts the commutator and its twin side by side. Any matrix product splits into two pieces:

$$AB = \underbrace{\tfrac{1}{2}(AB + BA)}_{\text{symmetric half}} + \underbrace{\tfrac{1}{2}(AB - BA)}_{\text{antisymmetric half}}.$$

The second is the commutator we know. The first has a name: the **anticommutator**,

$$\{A, B\} = AB + BA.$$

So $AB = \tfrac{1}{2}\{A,B\} + \tfrac{1}{2}[A,B]$. **The commutator and anticommutator are exactly the two halves of the product**, and they are distinguished by how they behave when you swap the two arguments.

Write $\sigma$ for the swap $A \leftrightarrow B$. Then:

| object | under the swap | name |
|---|---|---|
| $[A,B] = AB - BA$ | $[B,A] = -[A,B]$ | **odd** (antisymmetric): flips sign |
| $\{A,B\} = AB + BA$ | $\{B,A\} = \{A,B\}$ | **even** (symmetric): unchanged |

That is the entire content of "the commutator is swap-odd and the anticommutator is swap-even," and it follows from one line of arithmetic: reversing the order swaps which term is subtracted, and a difference flips sign while a sum does not.

The distinction has teeth whenever the *thing you are modeling* has a symmetry. If an experiment applies two interventions **simultaneously**, the observation carries no order, so it is invariant under the swap. An odd quantity cannot be read off such an observation: its sign is unobservable, because there is no "first" to compare against a "second." An even quantity can. Symmetry of the experiment decides which half of the algebra you are allowed to see.

## 7. The anticommutator is not a Lie object, and does not measure commuting

Two properties of $\{A,B\}$ are easy to state and important not to confuse with the commutator's.

**It does not vanish when the operators commute.** The most direct case is $B = A$:

$$[A, A] = 0 \qquad \text{but} \qquad \{A, A\} = A\cdot A + A \cdot A = 2A^2 \ne 0.$$

Everything commutes with itself, so the commutator is zero. The anticommutator is $2A^2$, which is as far from zero as $A$ is large. More generally, if $A$ and $B$ *do* commute, then $AB = BA$ and

$$\{A,B\} = AB + BA = 2AB,$$

which is not zero in general. So the anticommutator is **not** a test for commutativity, and it is **not** a measure of departure from additivity. Only the commutator is. Whatever $\lVert\{A,B\}\rVert$ measures, it is not "how non-additive is composing these two."

**It is not an operation of the Lie algebra.** Return to the skew-symmetric example of §5. If $A^\top = -A$ and $B^\top = -B$, then

$$\{A,B\}^\top = (AB + BA)^\top = B^\top A^\top + A^\top B^\top = BA + AB = \{A,B\}.$$

The anticommutator of two skew matrices is **symmetric**, the opposite of skew. It has left the algebra entirely. This is the precise sense in which the anticommutator is a different animal: the bracket keeps you inside the family of rotation generators, and the anticommutator ejects you from it. It belongs to a different structure, a **Jordan algebra**, which is built from the symmetric product rather than the antisymmetric one.

The consequence for modeling is direct: **the anticommutator never appears in BCH**, so it can never arise from composing operators. If you write it into a model, it is not being derived from the group product. It is a modeling choice you are making for some other reason, and you owe that reason.

## 8. So what does each one measure?

Both are bilinear pairings of two generators. They answer different questions, and the cleanest way to see it is to give each its own identity.

**The commutator is the correction to composing.** From BCH:

$$\log(\exp A \exp B) = A + B + \tfrac{1}{2}[A,B] + \cdots$$

It answers: *if I do $A$ and then $B$, how far do I land from where "$A$ and $B$ together" would have put me?* It is about **order and direction**. It vanishes precisely when composing is additive.

**The anticommutator is the cross-term of a square.** Expand:

$$(A+B)^2 = A^2 + AB + BA + B^2 = A^2 + \{A,B\} + B^2.$$

It answers: *if some quantity grows with the square of the total generator, how much do $A$ and $B$ reinforce or cancel each other?* It is the **interference term**, and it is about **magnitude and overlap**. Two generators pointing the same way have a large positive anticommutator; two pointing oppositely have a negative one. It says nothing about order, which is exactly right, since it is swap-even.

| | commutator $[A,B]$ | anticommutator $\{A,B\}$ |
|---|---|---|
| definition | $AB - BA$ | $AB + BA$ |
| under swap | odd (flips sign) | even (unchanged) |
| zero when they commute? | **yes, by definition** | **no** ($\{A,A\} = 2A^2$) |
| in the Lie algebra? | yes, closed | no, it is a Jordan product |
| appears in BCH? | yes, it *is* BCH | never |
| identity it lives in | $\log(e^Ae^B) = A + B + \tfrac12[A,B] + \cdots$ | $(A+B)^2 = A^2 + \{A,B\} + B^2$ |
| what it measures | order, direction, non-additivity | magnitude, overlap, interference |
| visible to a simultaneous experiment? | magnitude only, never its sign | yes |

Neither is "the right one." They are the odd and even halves of the same product, and which one you need is decided by which question you are asking and which symmetry your measurement has.

## 9. Where this goes

Three consequences carry into the rest of the corpus.

**Composition is one clean object.** Repeating a single operator is $\exp(M)^k = \exp(kM)$, one matrix rather than $k$ stacked network calls, which is the gallery's point about rollout. Composing *different* operators is $\exp(M_2)\exp(M_1)$, and BCH says the generator is $M_1 + M_2$ plus bracket corrections. Summing the generators inside one exponential, $\exp(M_1 + M_2)$, is the **commutative approximation**: correct when the operators commute, and wrong by the bracket series when they do not. Any model that aggregates a sequence of interventions by summing their coefficients has silently assumed they all commute.

**Non-commutativity is a modeling resource, not an obstacle.** The bracket is the one place an operator model can put an *interaction* between two interventions without adding a single parameter. It is a consequence of the two individual generators, not extra information fit from the pair.

**Symmetry decides what you can see.** A sequential experiment can observe the odd half. A simultaneous one cannot, and must be modeled with a swap-symmetric composition. Getting that right is not tidiness; it is the difference between a model that can be identified from your data and one that cannot.

---

*Previous: [A Gallery of Operators](02-operator-gallery.md). Symbols: [notation](notation.md). Where the algebra gets used: [Operator World Models](../operator_world_models/index.md) develops rollout and action-conditioned dynamics; the [GRL project](https://github.com/pleiadian53/GRL) is the deep reference for the operator formalism.*
