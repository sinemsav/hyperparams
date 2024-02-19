package encrypted

import (
	"encrypted_dbscan/comparator"
	"encrypted_dbscan/dbscan"
	"github.com/tuneinsight/lattigo/v4/ckks"
	"github.com/tuneinsight/lattigo/v4/dckks"
	"github.com/tuneinsight/lattigo/v4/rlwe"
	"github.com/tuneinsight/lattigo/v4/utils"
	"log"
	"math"
	"os"
)

// AggregateGrid aggregates the encrypted grids from clients and optionally performs comparison under encryption.
func AggregateGrid(clients []*dbscan.Client, inputType int, withComparison, dummyBootstrapping bool) []float64 {
	clients = createDummyClients(clients)
	//log.Println(len(clients))

	// Scheme params are taken directly from the proposed defaults
	paramsLiteral := ckks.PN14QP438
	if len(clients) >= 20 {
		paramsLiteral = ckks.PN15QP880
	}
	params, err := ckks.NewParametersFromLiteral(paramsLiteral) // PN13QP218 *PN14QP438* PN15QP880 PN16QP1761
	if err != nil {
		panic(err)
	}

	for _, client := range clients {
		client.GenerateKeyPair(params)
		client.PrepareInput(inputType)
	}

	crs, err := utils.NewKeyedPRNG([]byte{'f', 'e', 'd', 'd', 'b', 's', 'c', 'a', 'n'})
	if err != nil {
		panic(err)
	}

	// 1) Collective public key generation
	pk := ckgPhase(params, crs, clients)

	// 2) Collective relinearization key generation
	rlk := rkgPhase(params, crs, clients)

	// Encoder
	encoder := ckks.NewEncoder(params)
	encInputs, encryptor := encPhase(params, clients, pk, encoder)

	// Evaluator
	evaluator := ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: rlk, Rtks: nil})
	encRes := evalPhase(params, encInputs, evaluator)

	if !dummyBootstrapping {
		// Encryptor + Decryptor + Encoder
		dtb := &comparator.DebugToolbox{
			Encryptor: encryptor,
			Decryptor: nil,
			Encoder:   encoder,
			Evaluator: evaluator,
			CRS:       crs,
		}

		var ok bool
		if dtb.MinLevel, dtb.LogBound, ok = dckks.GetMinimumLevelForBootstrapping(128, params.DefaultScale(), len(clients), params.Q()); ok != true || dtb.MinLevel+1 > params.MaxLevel() {
			panic("Not enough levels to ensure correctness and 128 security")
		}

		RefreshParties := make([]*comparator.Party, len(clients))
		for i, client := range clients {
			p := new(comparator.Party)
			if i == 0 {
				p.RefreshProtocol = dckks.NewRefreshProtocol(params, dtb.LogBound, 3.2)
			} else {
				p.RefreshProtocol = RefreshParties[0].RefreshProtocol.ShallowCopy()
			}

			p.SK = client.MHEParams.SecKey
			p.Share = p.AllocateShare(dtb.MinLevel, params.MaxLevel())
			RefreshParties[i] = p
		}
		dtb.BootstrappingParties = RefreshParties

		// Threshold Comparison
		if withComparison {
			encRes = thresholdComparison(encRes, params, dtb)
		}
	}

	// Target private and public keys
	kgen := ckks.NewKeyGenerator(params)
	tsk, tpk := kgen.GenKeyPair()

	// Prepare for decryption via tsk
	encOut := pcksPhase(params, tpk, encRes, clients) // clients[0].MHEParams.pk

	if dummyBootstrapping {
		// Relinearization + rotation keys
		rlk = kgen.GenRelinearizationKey(tsk, 2)
		rots := []int{1, -1}
		rtks := kgen.GenRotationKeysForRotations(rots, false, tsk)

		// Encryptor + Decryptor + Encoder
		dtb := &comparator.DebugToolbox{
			Encryptor: ckks.NewEncryptor(params, tpk),
			Decryptor: ckks.NewDecryptor(params, tsk),
			Evaluator: ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: rlk, Rtks: rtks}),
			Encoder:   encoder,
		}
		// Threshold Comparison
		if withComparison {
			encOut = thresholdComparison(encOut, params, dtb)
		}
	}

	// Decrypt the result with the target secret key
	decryptor := ckks.NewDecryptor(params, tsk)
	//log.Println("> Result:")
	ptres := ckks.NewPlaintext(params, params.MaxLevel())
	decryptor.Decrypt(encOut, ptres)

	res := encoder.Decode(ptres, params.LogSlots())

	return dbscan.ComplexArrayToFloatArray(res, uint(dbscan.GRIDROWS*dbscan.GRIDCOLS))
}

// thresholdComparison performs the comparison algorithm and compares the ciphertext with the threshold.
func thresholdComparison(ciphertext *rlwe.Ciphertext, ckksParams ckks.Parameters, dtb *comparator.DebugToolbox) *rlwe.Ciphertext {
	threshold := float64(dbscan.DBSCAN_MINPOINTS) * dbscan.SCALEGRID

	// Comparison parameters
	n_f := 3 // Polynomial number for f
	n_g := 3 // Polynomial number for g
	df := 3  // Degree of composition of f
	dg := 3  // Degree of composition of g

	// Comparator
	comparator := comparator.NewComparator(ckksParams, dtb, uint(n_f), uint(n_g), uint(df), uint(dg))
	comparator.ProvideDebugToolBox(dtb)

	//fmt.Printf("\nIs v above a given threshold %v ?\n", complex(threshold, 0))

	return comparator.ThresholdNew(ciphertext, complex(threshold, 0))
}

// ckgPhase performs the Collective Public Key Generation phase and returns the public key.
func ckgPhase(params ckks.Parameters, crs utils.PRNG, clients []*dbscan.Client) *rlwe.PublicKey {
	l := log.New(os.Stderr, "", 0)
	l.Println("> CKG Phase")

	ckg := dckks.NewCKGProtocol(params) // Public key generation
	ckgCombined := ckg.AllocateShare()

	for _, client := range clients {
		client.MHEParams.CKGShare = ckg.AllocateShare()
	}

	crp := ckg.SampleCRP(crs)

	for _, client := range clients {
		ckg.GenShare(client.MHEParams.SecKey, crp, client.MHEParams.CKGShare)
	}

	pk := rlwe.NewPublicKey(params.Parameters)

	for _, client := range clients {
		ckg.AggregateShares(client.MHEParams.CKGShare, ckgCombined, ckgCombined)
	}
	ckg.GenPublicKey(ckgCombined, crp, pk)

	return pk
}

// rkgPhase performs the Collective Relinearization Key Generation phase and returns the relineartization key.
func rkgPhase(params ckks.Parameters, crs utils.PRNG, clients []*dbscan.Client) *rlwe.RelinearizationKey {
	l := log.New(os.Stderr, "", 0)
	l.Println("> RKG Phase")

	rkg := dckks.NewRKGProtocol(params) // Relineariation key generation
	_, rkgCombined1, rkgCombined2 := rkg.AllocateShare()

	for _, client := range clients {
		client.MHEParams.RLKEphemSecKey, client.MHEParams.RKGShareOne, client.MHEParams.RKGShareTwo = rkg.AllocateShare()
	}

	crp := rkg.SampleCRP(crs)

	for _, client := range clients {
		rkg.GenShareRoundOne(client.MHEParams.SecKey, crp, client.MHEParams.RLKEphemSecKey, client.MHEParams.RKGShareOne)
	}

	for _, client := range clients {
		rkg.AggregateShares(client.MHEParams.RKGShareOne, rkgCombined1, rkgCombined1)
	}

	for _, client := range clients {
		rkg.GenShareRoundTwo(client.MHEParams.RLKEphemSecKey, client.MHEParams.SecKey, rkgCombined1, client.MHEParams.RKGShareTwo)
	}

	rlk := rlwe.NewRelinearizationKey(params.Parameters, 1)

	for _, client := range clients {
		rkg.AggregateShares(client.MHEParams.RKGShareTwo, rkgCombined2, rkgCombined2)
	}
	rkg.GenRelinearizationKey(rkgCombined1, rkgCombined2, rlk)

	return rlk
}

// encPhase creates the encryptor given the public key and encrypts the clients' inputs.
func encPhase(params ckks.Parameters, clients []*dbscan.Client, pk *rlwe.PublicKey, encoder ckks.Encoder) (encInputs []*rlwe.Ciphertext, encryptor rlwe.Encryptor) {
	l := log.New(os.Stderr, "", 0)

	encInputs = make([]*rlwe.Ciphertext, len(clients))
	for i := range encInputs {
		encInputs[i] = ckks.NewCiphertext(params, 1, params.MaxLevel())
	}

	// Each party encrypts its input vector
	l.Println("> Encrypt Phase")
	encryptor = ckks.NewEncryptor(params, pk)

	pt := ckks.NewPlaintext(params, params.MaxLevel())
	for i, client := range clients {
		encoder.Encode(client.MHEParams.InputOne, pt, params.LogSlots())
		encryptor.Encrypt(pt, encInputs[i])
	}

	return
}

// evalPhase evaluates the sum of the encrypted values
func evalPhase(params ckks.Parameters, encInputs []*rlwe.Ciphertext, evaluator ckks.Evaluator) (encRes *rlwe.Ciphertext) {
	//l := log.New(os.Stderr, "", 0)

	encLvls := make([][]*rlwe.Ciphertext, 0)
	encLvls = append(encLvls, encInputs)
	for nLvl := len(encInputs) / 2; nLvl > 0; nLvl = nLvl >> 1 {
		encLvl := make([]*rlwe.Ciphertext, nLvl)
		for i := range encLvl {
			encLvl[i] = ckks.NewCiphertext(params, 2, params.MaxLevel())
		}
		encLvls = append(encLvls, encLvl)
	}
	encRes = encLvls[len(encLvls)-1][0]

	for i, lvl := range encLvls[:len(encLvls)-1] {
		nextLvl := encLvls[i+1]
		//l.Println("\tlevel", i, len(lvl), "->", len(nextLvl))

		for j, nextLvlCt := range nextLvl {
			evaluator.Add(lvl[2*j], lvl[2*j+1], nextLvlCt)
		}
	}

	return
}

// pcksPhase runs the Collective Key Switching Protocol from the collective secret key to the target public key.
func pcksPhase(params ckks.Parameters, tpk *rlwe.PublicKey, encRes *rlwe.Ciphertext, clients []*dbscan.Client) (encOut *rlwe.Ciphertext) {
	l := log.New(os.Stderr, "", 0)

	// Collective key switching from the collective secret key to the target public key
	pcks := dckks.NewPCKSProtocol(params, 3.19)

	for _, client := range clients {
		client.MHEParams.PCKSShare = pcks.AllocateShare(params.MaxLevel())
	}

	l.Println("> PCKS Phase")
	for _, client := range clients {
		pcks.GenShare(client.MHEParams.SecKey, tpk, encRes, client.MHEParams.PCKSShare)
	}

	pcksCombined := pcks.AllocateShare(params.MaxLevel())
	encOut = ckks.NewCiphertext(params, 1, params.MaxLevel())

	for _, client := range clients {
		pcks.AggregateShares(client.MHEParams.PCKSShare, pcksCombined, pcksCombined)
	}
	pcks.KeySwitch(encRes, pcksCombined, encOut)

	return
}

// createDummyClients creates additional clients in order to have a power of two number of clients for easier evaluation of encrypted inputs.
// In case of 10 clients this function will append 6 dummy clients, and in case of 20 clients the function appends 12 dummy clients.
func createDummyClients(clients []*dbscan.Client) []*dbscan.Client {
	// Find the smallest power of two greater than len(clients)
	power := int(math.Ceil(math.Log2(float64(len(clients)))))
	numPlaceHolders := int(math.Pow(2, float64(power))) - len(clients)

	dummy := make([]*dbscan.Client, numPlaceHolders)
	for i := 0; i < numPlaceHolders; i++ {
		dummy[i] = &dbscan.Client{}
		dummy[i].CreateEmptyClient()
	}
	return append(clients, dummy...)
}
