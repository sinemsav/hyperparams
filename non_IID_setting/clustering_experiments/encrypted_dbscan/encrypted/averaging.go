package encrypted

import (
	"encrypted_dbscan/comparator"
	"encrypted_dbscan/dbscan"
	"github.com/tuneinsight/lattigo/v4/ckks"
	"github.com/tuneinsight/lattigo/v4/rlwe"
	"github.com/tuneinsight/lattigo/v4/utils"
	"log"
	"os"
)

// AvgEncrypted is used to compute average of clients params (acc/lr/mom) under encryption.
func AvgEncrypted(clients []*dbscan.Client, inputType int) []float64 {
	clients = createDummyClients(clients)

	// Scheme params are taken directly from the proposed defaults
	params, err := ckks.NewParametersFromLiteral(ckks.PN14QP438) // PN13QP218 PN14QP438 PN15QP880 PN16QP1761
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

	pk := ckgPhase(params, crs, clients)
	rlk := rkgPhase(params, crs, clients)

	// Encoding
	encoder := ckks.NewEncoder(params)

	encInputsAcc, encInputsNum := encPhaseAVG(params, clients, pk, encoder)

	// Evaluation
	evaluator := ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: rlk, Rtks: nil})
	encResAcc := evalPhase(params, encInputsAcc, evaluator)
	encResNum := evalPhase(params, encInputsNum, evaluator)

	// Inverse
	encRes, err := evaluator.InverseNew(encResNum, 7)
	if err != nil {
		panic(err)
	}

	// Multiplication
	encRes = evaluator.MulRelinNew(encRes, encResAcc)

	// Target private and public keys
	kgen := ckks.NewKeyGenerator(params)
	tsk, tpk := kgen.GenKeyPair()
	dtb := &comparator.DebugToolbox{
		Encryptor: ckks.NewEncryptor(params, tpk),
		Decryptor: ckks.NewDecryptor(params, tsk),
		Encoder:   encoder,
	}

	// Prepare for decryption via tsk (key switch to tpk)
	encOutRes := pcksPhase(params, tpk, encRes, clients)

	// Decrypt the result with the target secret key
	pt := dtb.Decryptor.DecryptNew(encOutRes)

	// Decode
	res := encoder.Decode(pt, params.LogSlots())

	return dbscan.ComplexArrayToFloatArray(res, uint(dbscan.GRIDROWS*dbscan.GRIDCOLS))
}

// encPhaseAVG encrypts two sets of inputs that will be divided to compute average - the parameter values (inside cluster) and the cluster size.
func encPhaseAVG(params ckks.Parameters, clients []*dbscan.Client, pk *rlwe.PublicKey, encoder ckks.Encoder) (encInputs1, encInputs2 []*rlwe.Ciphertext) {
	l := log.New(os.Stderr, "", 0)
	l.Println("> ENC Phase")

	encInputs1 = make([]*rlwe.Ciphertext, len(clients))
	encInputs2 = make([]*rlwe.Ciphertext, len(clients))
	for i := range encInputs1 {
		encInputs1[i] = ckks.NewCiphertext(params, 1, params.MaxLevel())
		encInputs2[i] = ckks.NewCiphertext(params, 1, params.MaxLevel())
	}

	// Each party encrypts its input vector
	encryptor := ckks.NewEncryptor(params, pk)

	pt1 := ckks.NewPlaintext(params, params.MaxLevel())
	pt2 := ckks.NewPlaintext(params, params.MaxLevel())
	for i, client := range clients {
		encoder.Encode(client.MHEParams.InputOne, pt1, params.LogSlots())
		encoder.Encode(client.MHEParams.InputTwo, pt2, params.LogSlots())
		encryptor.Encrypt(pt1, encInputs1[i])
		encryptor.Encrypt(pt2, encInputs2[i])
	}

	l.Printf("\tdone encrypt")

	return
}
