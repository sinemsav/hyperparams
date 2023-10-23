package comparator

import (
	"fmt"
	"strings"

	"github.com/tuneinsight/lattigo/v4/ckks"
	"github.com/tuneinsight/lattigo/v4/rlwe"
)

// Allow to have more verbose output
const Verbose = false
const NbValToShow = 5

type Comparator interface {
	Compare(a, b *rlwe.Ciphertext, n_f, n_g uint, df, dg uint, ctOut *rlwe.Ciphertext)
	CompareNew(a, b *rlwe.Ciphertext, n_f, n_g uint, df, dg uint) *rlwe.Ciphertext
}

type comparator struct {
	params    ckks.Parameters
	eval      ckks.Evaluator
	encoder   ckks.Encoder
	encryptor rlwe.Encryptor
	nf, ng    uint
	df, dg    uint
	dtb       *DebugToolbox
}

func NewComparator(params ckks.Parameters, dtb *DebugToolbox, n_f, n_g uint, df, dg uint) comparator {
	return comparator{
		params:    params,
		encoder:   dtb.Encoder,   //ckks.NewEncoder(params),
		encryptor: dtb.Encryptor, //ckks.NewEncryptor(params, pk),
		eval:      dtb.Evaluator, //ckks.NewEvaluator(params, evaluationKey),
		nf:        n_f,
		ng:        n_g,
		df:        df,
		dg:        dg,
		dtb:       nil,
	}
}

func (c *comparator) ProvideDebugToolBox(_dtb *DebugToolbox) {
	c.dtb = _dtb
}

/**************************************************************************\
| 													Comparison methods														 |
\**************************************************************************/

// Compare implements Comparator
func (c *comparator) Compare(a, b, ctOut *rlwe.Ciphertext) {

	result := c.CompareNew(a, b)

	// Copy the result into the provided ciphertext
	ctOut.Copy(result)
}

// CompareNew implements Comparator
func (c *comparator) CompareNew(a, b *rlwe.Ciphertext) *rlwe.Ciphertext {
	fn := GetPolynomialF(c.nf)
	gn := GetPolynomialG(c.ng)

	if Verbose {
		if 0 < c.df {
			fmt.Printf("Chosen Polynomial f%v: %v (depth %v)\n", c.nf, repr_poly(fn), fn.Depth())
		}
		if 0 < c.dg {
			fmt.Printf("Chosen Polynomial g%v: %v (depth %v)\n", c.ng, repr_poly(gn), gn.Depth())
		}
		fmt.Printf("Ciphertext a of depth: %v\n", a.Level())
		fmt.Printf("Ciphertext b of depth: %v\n\n", b.Level())
	}

	// Get a-b
	result := c.eval.SubNew(a, b)

	// Evaluate the difference in g
	if Verbose {
		fmt.Printf("Evaluate composition for g_%v\n", c.ng)
	}
	result = c.composePolyNew(result, gn, c.dg)

	// Evaluate the difference in f
	if Verbose {
		fmt.Printf("Evaluate composition for f_%v\n", c.nf)
	}
	result = c.composePolyNew(result, fn, c.df)

	// Ensure enough depth
	result = c.bootstrapIfNeeded(result, gn)

	if Verbose {
		fmt.Printf("Before moving [-1, 1] -> [0, 1]\n")
		c.printDebug(result)
	}

	// Add 1 and divide by 2 to output result in [0,1]
	c.eval.AddConst(result, 1, result)
	c.eval.MultByConst(result, 0.5, result)

	return result
}

/**************************************************************************\
| 														Threshold methods														 |
|																																					 |
| 	Could use ReLu to map 1/2 (if value is exactly the threshold) to 1		 |
\**************************************************************************/

func (c *comparator) ThresholdNew(ct *rlwe.Ciphertext, threshold complex128) *rlwe.Ciphertext {
	thresholdvals := make([]complex128, c.params.Slots())
	for i := 0; i < c.params.Slots(); i++ {
		thresholdvals[i] = threshold
	}
	thresholdEnc := c.encoder.EncodeNew(thresholdvals, c.params.MaxLevel(), c.params.DefaultScale(), c.params.LogSlots())
	thresholdCT := c.encryptor.EncryptNew(thresholdEnc)
	return c.CompareNew(ct, thresholdCT)
}

/**************************************************************************\
| 														Utility functions														 |
\**************************************************************************/

func (c *comparator) composePolyNew(ct *rlwe.Ciphertext, poly *ckks.Polynomial, dn uint) *rlwe.Ciphertext {
	out := ct.CopyNew()

	for deg := uint(0); deg < dn; deg++ {
		if Verbose {
			fmt.Printf("Composition %v/%v\n", deg, dn-1)
		}

		// Ensure enough depth
		outBootstrapped := c.bootstrapIfNeeded(out, poly)

		// Evaluate with the provided polynomial
		evaluatedPoly, err := c.eval.EvaluatePoly(outBootstrapped, poly, c.params.DefaultScale()) //outBootstrapped.Scale)
		if nil != err {
			fmt.Printf("Error while EvaluatePoly for polynomial %v (composition number %v/%v): %v\n", repr_poly(poly), deg, dn-1, err)
			return nil
		}
		c.printDebug(evaluatedPoly)
		out = evaluatedPoly
	}
	return out
}

func enoughLevels(levels, bootstrapMinLevel int, pol *ckks.Polynomial) bool {
	//fmt.Printf("currentLevel = %d \t minBootstrapLevel = %d \t polyDepth = %d \n", levels, bootstrapMinLevel, pol.Depth())
	return !(levels < pol.Depth() || (levels-pol.Depth()) < bootstrapMinLevel)
}

func (c *comparator) bootstrapIfNeeded(ct *rlwe.Ciphertext, pol *ckks.Polynomial) *rlwe.Ciphertext {
	if !enoughLevels(ct.Level(), c.dtb.MinLevel, pol) {

		if c.dtb == nil {
			fmt.Println("Unable to perform dummy bootstrap. Missing DebugToolBox")
			return ct
		}

		// Bootstrapping process
		if c.dtb.Decryptor != nil {
			// Dummy bootstrapping
			decrypted := c.dtb.Decryptor.DecryptNew(ct)
			decryptedValues := c.dtb.Encoder.Decode(decrypted, c.params.LogSlots())
			plaintext := c.dtb.Encoder.EncodeNew(decryptedValues, c.params.MaxLevel(), c.params.DefaultScale(), c.params.LogSlots())
			ct = c.dtb.Encryptor.EncryptNew(plaintext)

			// fmt.Printf("Dummy bootstrap done. New ciphertext level: %v\n", ct.Level())
		} else {

			//fmt.Printf("Not Dummy bootstrap > \n")
			//fmt.Printf("\t\tDropping %d level(s) to get to level %d (=minLevel+1)\n", ct.Level()-c.dtb.MinLevel-1, c.dtb.MinLevel+1)

			c.dtb.Evaluator.DropLevel(ct, ct.Level()-c.dtb.MinLevel-1)

			P0 := c.dtb.BootstrappingParties[0]
			crp := P0.SampleCRP(c.params.MaxLevel(), c.dtb.CRS)

			for i, p := range c.dtb.BootstrappingParties {
				p.GenShare(p.SK, c.dtb.LogBound, c.params.LogSlots(), ct, crp, p.Share)
				if i > 0 {
					P0.AggregateShares(p.Share, P0.Share, P0.Share)
				}
			}

			P0.Finalize(ct, c.params.LogSlots(), crp, P0.Share, ct)

			//fmt.Printf("\t\tNew ciphertext level: %v\n", ct.Level())

		}

	}
	return ct
}

func (c *comparator) printDebug(ct *rlwe.Ciphertext) {
	if Verbose && nil != c.dtb {
		c.printDebugVerbose(ct)
	}
}

func (c *comparator) printDebugVerbose(ct *rlwe.Ciphertext) {
	decrypted := c.dtb.Decryptor.DecryptNew(ct)
	decryptedValues := c.dtb.Encoder.Decode(decrypted, c.params.LogSlots())
	fmt.Printf("Current values : ")
	for i := 0; i < NbValToShow; i++ {
		fmt.Printf("%f ", processBeforePrint(decryptedValues[i]))
	}
	fmt.Printf("\nRemaining levels: %v\tDegree: %v\tScale: %v\n", ct.Level(), ct.Degree(), ct.Scale)
}

func repr_poly(p *ckks.Polynomial) string {
	builder := strings.Builder{}

	for i := len(p.Coeffs); 0 < i; i-- {
		c := real(p.Coeffs[i-1])
		c_negativ := c < 0
		if c == 0 {
			continue
		}

		if i != len(p.Coeffs) {
			builder.WriteString(" ")
		}

		if c_negativ {
			c = -c
			builder.WriteString("-")
		}

		if i != len(p.Coeffs) {
			if !c_negativ {
				builder.WriteString("+")
			}
			builder.WriteString(" ")
		}

		builder.WriteString(fmt.Sprintf("%.3f", c))
		builder.WriteString("x")

		if deg := i - 1; 1 < deg {
			builder.WriteString("^")
			builder.WriteString(fmt.Sprintf("%v", deg))
		}
	}

	return builder.String()
}
